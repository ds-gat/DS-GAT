import os
import csv

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

input_files = [
    "train.tsv",
    "val.tsv",
    "test.tsv",
  #  "softlogic.tsv",
  #  "test_with_neg.tsv"
]

output_suffix = "_id"  # train_id.tsv etc.


# ─────────────────────────────────────────────────────────────
# STEP 1 — COLLECT UNIQUE ENTITIES & RELATIONS
# ─────────────────────────────────────────────────────────────

entities = set()
relations = set()

for filename in input_files:
    if not os.path.exists(filename):
        continue

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue

            head, relation, tail = parts[:3]

            entities.add(head)
            entities.add(tail)
            relations.add(relation)

print(f"Total unique entities: {len(entities)}")
print(f"Total unique relations: {len(relations)}")


# ─────────────────────────────────────────────────────────────
# STEP 2 — CREATE ID MAPPINGS
# ─────────────────────────────────────────────────────────────

entity2id = {entity: idx for idx, entity in enumerate(sorted(entities))}
relation2id = {rel: idx for idx, rel in enumerate(sorted(relations))}


# ─────────────────────────────────────────────────────────────
# STEP 3 — SAVE entity_id.csv
# ─────────────────────────────────────────────────────────────

with open("entity_id.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["entity string", "id"])
    for entity, idx in entity2id.items():
        writer.writerow([entity, idx])

print("Saved entity_id.csv")


# ─────────────────────────────────────────────────────────────
# STEP 4 — SAVE relation_id.csv
# ─────────────────────────────────────────────────────────────

with open("relation_id.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["relation string", "id"])
    for rel, idx in relation2id.items():
        writer.writerow([rel, idx])

print("Saved relation_id.csv")


# ─────────────────────────────────────────────────────────────
# STEP 5 — REWRITE DATA FILES WITH IDs
# ─────────────────────────────────────────────────────────────

for filename in input_files:
    if not os.path.exists(filename):
        continue

    output_file = filename.replace(".tsv", f"{output_suffix}.tsv")

    with open(filename, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for line in fin:
            parts = line.strip().split("\t")

            if len(parts) < 3:
                continue

            head, relation, tail = parts[:3]
            rest = parts[3:]  # confidence or label etc.

            h_id = entity2id[head]
            r_id = relation2id[relation]
            t_id = entity2id[tail]

            new_line = [str(h_id), str(r_id), str(t_id)] + rest
            fout.write("\t".join(new_line) + "\n")

    print(f"Saved {output_file}")

print("\nAll files processed successfully.")