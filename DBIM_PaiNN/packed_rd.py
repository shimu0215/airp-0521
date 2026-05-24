import argparse
import bisect
import glob
import json
import os
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data


PACKED_RD_VERSION = "packed_rd_v1"
PACKED_RD_INDEX = "packed_rd_index.json"


def _to_python_int(value):
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def _slice_real_atoms(value, natoms):
    if isinstance(value, torch.Tensor):
        return value[:natoms].detach().cpu()
    raise TypeError(f"Expected tensor field, got {type(value)!r}")


def _pack_single_file(input_path, output_path):
    sample_list = torch.load(input_path, weights_only=False, map_location="cpu")
    if not isinstance(sample_list, list):
        raise TypeError(f"Expected a list of Data objects in {input_path}")
    if not sample_list:
        raise ValueError(f"Empty data file: {input_path}")

    pos_chunks = []
    rdkit_pos_chunks = []
    atomic_number_chunks = []
    force_chunks = []
    npa_chunks = []
    atom_offsets = [0]
    natoms_list = []
    energy_list = []
    species_list = []
    source_batch_ids = []
    smiles_list = []

    max_atom_number = None
    max_atom_id = 0

    for molecule_idx, sample in enumerate(sample_list):
        natoms = _to_python_int(sample.natoms)
        if natoms <= 0:
            raise ValueError(f"Invalid natoms={natoms} in {input_path} at sample {molecule_idx}")

        max_atom_number = sample.pos.shape[0] if max_atom_number is None else max_atom_number
        max_atom_id = max(max_atom_id, int(sample.atomic_numbers[:natoms].max().item()) + 1)

        pos = _slice_real_atoms(sample.pos, natoms).to(torch.float32)
        rdkit_pos = _slice_real_atoms(sample.rdkit_pos, natoms).to(torch.float32)
        atomic_numbers = _slice_real_atoms(sample.atomic_numbers, natoms).to(torch.int16)
        energy_grad = _slice_real_atoms(sample.energy_grad, natoms).to(torch.float32)
        npa_charges = _slice_real_atoms(sample.npa_charges, natoms).to(torch.float32)

        species_field = sample.species
        if isinstance(species_field, torch.Tensor) and species_field.ndim > 0:
            species_value = int(species_field[:natoms][0].item())
        else:
            species_value = _to_python_int(species_field)

        batch_field = getattr(sample, "batch", None)
        if isinstance(batch_field, torch.Tensor) and batch_field.numel() > 0:
            source_batch = int(batch_field[:natoms][0].item())
        else:
            source_batch = molecule_idx

        pos_chunks.append(pos)
        rdkit_pos_chunks.append(rdkit_pos)
        atomic_number_chunks.append(atomic_numbers)
        force_chunks.append(energy_grad)
        npa_chunks.append(npa_charges)
        natoms_list.append(natoms)
        energy_list.append(sample.energy.detach().cpu().reshape(1).to(torch.float32))
        species_list.append(species_value)
        source_batch_ids.append(source_batch)
        smiles_list.append(getattr(sample, "smiles", ""))
        atom_offsets.append(atom_offsets[-1] + natoms)

    packed = {
        "format": PACKED_RD_VERSION,
        "source_file": os.path.basename(input_path),
        "num_molecules": len(sample_list),
        "max_atom_number": int(max_atom_number),
        "max_atom_id": int(max_atom_id),
        "atom_offsets": torch.tensor(atom_offsets, dtype=torch.int64),
        "natoms": torch.tensor(natoms_list, dtype=torch.int16),
        "energy": torch.cat(energy_list, dim=0),
        "species": torch.tensor(species_list, dtype=torch.int8),
        "source_batch_ids": torch.tensor(source_batch_ids, dtype=torch.int32),
        "smiles": smiles_list,
        "pos": torch.cat(pos_chunks, dim=0),
        "rdkit_pos": torch.cat(rdkit_pos_chunks, dim=0),
        "atomic_numbers": torch.cat(atomic_number_chunks, dim=0),
        "energy_grad": torch.cat(force_chunks, dim=0),
        "npa_charges": torch.cat(npa_chunks, dim=0),
    }
    torch.save(packed, output_path)

    return {
        "path": os.path.basename(output_path),
        "source_file": os.path.basename(input_path),
        "num_molecules": len(sample_list),
        "max_atom_number": int(max_atom_number),
        "max_atom_id": int(max_atom_id),
    }


def pack_rd_directory(input_dir, output_dir, pattern="qm9star_*_chunk*_processed.pt", overwrite=False):
    os.makedirs(output_dir, exist_ok=True)
    input_files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not input_files:
        raise FileNotFoundError(f"No files matched {pattern!r} under {input_dir}")

    index = {
        "format": PACKED_RD_VERSION,
        "input_dir": os.path.abspath(input_dir),
        "output_dir": os.path.abspath(output_dir),
        "files": [],
    }

    for input_path in input_files:
        base_name = os.path.basename(input_path).replace("_processed.pt", "_packed.pt")
        output_path = os.path.join(output_dir, base_name)
        if os.path.exists(output_path) and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {output_path}")

        print(f"Packing {input_path} -> {output_path}")
        file_meta = _pack_single_file(input_path, output_path)
        index["files"].append(file_meta)

    index["total_molecules"] = sum(item["num_molecules"] for item in index["files"])
    index_path = os.path.join(output_dir, PACKED_RD_INDEX)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print(f"Saved packed index to {index_path}")
    return index


class PackedRDChunkDataset(Dataset):
    def __init__(self, data_dir, cache_size=2):
        self.data_dir = os.path.abspath(data_dir)
        self.cache_size = max(1, int(cache_size))
        self.index_path = os.path.join(self.data_dir, PACKED_RD_INDEX)
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Packed RD index not found: {self.index_path}")

        with open(self.index_path, "r", encoding="utf-8") as f:
            self.index = json.load(f)

        if self.index.get("format") != PACKED_RD_VERSION:
            raise ValueError(f"Unsupported packed RD format: {self.index.get('format')}")

        self.files = self.index["files"]
        self.cumulative_sizes = []
        running_total = 0
        for item in self.files:
            running_total += int(item["num_molecules"])
            self.cumulative_sizes.append(running_total)

        self.chunk_cache = OrderedDict()

    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def _load_chunk(self, chunk_idx):
        if chunk_idx in self.chunk_cache:
            chunk = self.chunk_cache.pop(chunk_idx)
            self.chunk_cache[chunk_idx] = chunk
            return chunk

        chunk_path = os.path.join(self.data_dir, self.files[chunk_idx]["path"])
        chunk = torch.load(chunk_path, weights_only=False, map_location="cpu")
        self.chunk_cache[chunk_idx] = chunk

        while len(self.chunk_cache) > self.cache_size:
            self.chunk_cache.popitem(last=False)

        return chunk

    def _make_padded_tensor(self, values, natoms, max_atom_number, trailing_shape):
        padded_shape = (max_atom_number,) + trailing_shape
        padded = values.new_zeros(padded_shape)
        padded[:natoms] = values
        return padded

    def _reconstruct_sample(self, chunk, local_idx):
        offsets = chunk["atom_offsets"]
        start = int(offsets[local_idx].item())
        end = int(offsets[local_idx + 1].item())
        natoms = int(chunk["natoms"][local_idx].item())
        max_atom_number = int(chunk["max_atom_number"])
        max_atom_id = int(chunk["max_atom_id"])

        pos = self._make_padded_tensor(chunk["pos"][start:end], natoms, max_atom_number, (3,))
        rdkit_pos = self._make_padded_tensor(chunk["rdkit_pos"][start:end], natoms, max_atom_number, (3,))
        atomic_numbers = self._make_padded_tensor(
            chunk["atomic_numbers"][start:end].to(torch.long), natoms, max_atom_number, ()
        )
        energy_grad = self._make_padded_tensor(chunk["energy_grad"][start:end], natoms, max_atom_number, (3,))
        npa_charges = self._make_padded_tensor(chunk["npa_charges"][start:end], natoms, max_atom_number, ())

        atom_mask = torch.zeros((max_atom_number, 1), dtype=torch.float32)
        atom_mask[:natoms] = 1.0

        edge_mask = torch.zeros((max_atom_number, max_atom_number), dtype=torch.float32)
        edge_mask[:natoms, :natoms] = 1.0

        species_value = int(chunk["species"][local_idx].item())
        species = torch.zeros(max_atom_number, dtype=torch.long)
        species[:natoms] = species_value

        batch_value = int(chunk["source_batch_ids"][local_idx].item())
        batch = torch.zeros(max_atom_number, dtype=torch.long)
        batch[:natoms] = batch_value

        atomic_numbers_one_hot = torch.zeros((max_atom_number, max_atom_id), dtype=torch.float32)
        atomic_numbers_one_hot[:natoms] = F.one_hot(
            atomic_numbers[:natoms], num_classes=max_atom_id
        ).to(torch.float32)

        return Data(
            pos=pos,
            atomic_numbers=atomic_numbers,
            atomic_numbers_one_hot=atomic_numbers_one_hot,
            atom_mask=atom_mask,
            edge_mask=edge_mask,
            batch=batch,
            natoms=torch.tensor(natoms, dtype=torch.long),
            energy=chunk["energy"][local_idx].clone(),
            energy_grad=energy_grad,
            npa_charges=npa_charges,
            species=species,
            smiles=chunk["smiles"][local_idx],
            rdkit_pos=rdkit_pos,
        )

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        chunk_idx = bisect.bisect_right(self.cumulative_sizes, index)
        prev_total = 0 if chunk_idx == 0 else self.cumulative_sizes[chunk_idx - 1]
        local_idx = index - prev_total
        chunk = self._load_chunk(chunk_idx)
        return self._reconstruct_sample(chunk, local_idx)


def main():
    parser = argparse.ArgumentParser(description="Convert expanded data_rd files into packed chunk files.")
    parser.add_argument("--input_dir", required=True, help="Directory containing qm9star_*_processed.pt list files.")
    parser.add_argument("--output_dir", required=True, help="Directory to write packed RD files into.")
    parser.add_argument(
        "--pattern",
        default="qm9star_*_chunk*_processed.pt",
        help="Glob pattern for input files.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing packed files.")
    args = parser.parse_args()

    pack_rd_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pattern=args.pattern,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
