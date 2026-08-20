# XMark pre/post/level Parquet Generator

This folder contains the XMark benchmark XML generator and a script that turns its
output into pre/post/level tree-position ids, as used by pre/post ("XPath
Accelerator") tree indexes over XML/JSON semi-structured data.

## 1. Generating the XML corpus

`xmlgen.Linux` is the standard XMark benchmark generator (Schmidt/Waas et al.,
*XMark: A Benchmark for XML Data Management*). It is a 32-bit Linux binary and ships
with no bundled docs; its `--help` usage string is:

```
Usage: xmlgen [ -h ] [ -d|i|t|v|e ] [ -f <factor> ] [ -o <file> ] [ -s <cnt> ]
```

The relevant flag is `-f <factor>`, XMark's own scaling convention: a factor of
`1.0` produces ~100 MB of XML, and it scales linearly, so `10.0` produces ~1 GB:

```bash
chmod +x xmlgen.Linux   # ships without the execute bit
./xmlgen.Linux -f 10.0 -o xmark_10.0.xml
```

This repo's copy was generated exactly this way; the resulting `xmark_10.0.xml` is
1.1 GB. `xmark_10.0.xml` is not committed (see "Not committed" below).

## 2. Generating the pre/post/level ids

```bash
python generate_prepost_ids.py --xml xmark_10.0.xml --mode elements
python generate_prepost_ids.py --xml xmark_10.0.xml --mode full
```

### Node model (`--mode`)

There is no single node model in the pre/post literature — Grust's XPath Accelerator
numbers every "database node" (elements, attributes, text, comments, PIs) in one
shared pre/post space, while most production pre/post stores (BaseX, MonetDB/XQuery)
only number elements and treat attributes as element properties. Both are generated
here:

- **`elements`** — only element nodes get a (level, pre, post) triple.
- **`full`** — elements, attributes, and text nodes (both the text immediately
  inside an element and the "tail" text between a child's closing tag and the next
  sibling, i.e. XMark's mixed-content `<text>...<keyword>...</keyword>...</text>`
  spans) are all numbered in one shared pre/post space, per Grust. Attribute and text
  leaf nodes are ordered immediately at the point they occur in document order — an
  element's attributes (in document order) come right after the element's own `pre`,
  before its first child or text.

On the real 1.1 GB corpus (`xmark_10.0.xml`, factor 10.0), from the generator's own
run: 16,703,210 nodes under `elements` (max level 11, 0-indexed from the root); under
`full`, 32,298,988 nodes total — 16,703,210 elements, 3,829,768 attributes, and
11,766,010 non-whitespace text spans (leading + tail combined) — max level 12 (text
and attribute leaves sit one level below their owning element).

### Traversal (iterative, no Python recursion)

The generator uses `xml.etree.ElementTree.iterparse(path, events=("start","end"))`,
which streams `start`/`end` callbacks from the C `expat` parser — the file is never
loaded as a DOM and the traversal never recurses in Python; nesting is tracked with an
explicit list used as a stack, and each element is `.clear()`-ed once nothing further
is needed from it, so memory stays bounded on multi-GB input.

One correctness subtlety in `full` mode: `Element.tail` (text after a child's closing
tag) is only guaranteed populated once parsing has moved past it — i.e. at the next
sibling's `start` event or the parent's own `end` event, never reliably at the child's
own `end` event. The generator defers reading (and clearing) a closed child until that
point. `Element.text` doesn't have this issue — it's already final by the time the
first child's `start` event fires.

## 3. Bit-packed id (`prepost_ids_<mode>.parquet`, column `prepost_id`, int64)

There is no fixed-width literature standard for packing pre/post/level into one
integer (ORDPATH/BIRD are variable-length self-describing labels; XPath Accelerator
stores pre/post/level as separate relational columns, not packed). The split below is
sized to the real corpus above with headroom, as named constants in
`generate_prepost_ids.py` so it's trivial to widen for a larger scale factor:

| Field | Bits | Range | Why |
|---|---|---|---|
| `level` | 8 (bits 63–56) | 0–255 | observed max level 12 (full mode); ~20x headroom |
| `pre`   | 28 (bits 55–28) | 0–268,435,455 | observed max node count 32.3M (full); ~8x headroom |
| `post`  | 28 (bits 27–0)  | 0–268,435,455 | same as `pre` |

`level` sits in the most-significant bits (separates levels into disjoint numeric
bands); `pre` sits above `post` because `pre` correlates with document/insertion
order and is the more delta/FOR-friendly field — relevant since this dataset exists
to stress-test SubIntSplit's per-section bit-splitting DP against a 3-field composite
key with very different per-field statistics than Twitter Snowflake ids.

The packed value is built as `uint64` then bit-reinterpreted to `int64` via
`.view(np.int64)`, matching the convention already used by
`../IPv4/create_ipv4_ids_parquet.py` and
`../TwitterSnowflake/create_twitter_snowflake_parquet.py`. The generator asserts each
field is within its bit budget before packing and raises a clear error (naming the
offending field) rather than silently truncating.

## 4. Unpacked columns (one file per field, int32)

For studying compression the way a column store (e.g. MonetDB) actually lays out
pre/post/level — one column per field, not a packed key — the generator also writes:

- `level_<mode>.parquet` — column `level`, int32
- `pre_<mode>.parquet` — column `pre`, int32
- `post_<mode>.parquet` — column `post`, int32

All three fields fit comfortably in 32 bits on the observed corpus (matching the
28-bit budgets above), bit-reinterpreted `uint32 -> int32` the same way.

These are not yet wired into `Source/benchmark/registry/DatasetRegistry.hpp`, which
only defines `int64Datasets()` today — its own doc comment already flags int32
columns (TPCH orderkey, iNaturalist) as needing a future `int32Datasets()` plus an
int32 encoder registry. Registering these three columns is a one-line addition to
that future work.

## 5. Not committed

`Datasets/` is not tracked in this repository (confirmed: `tweet_ids.parquet`, 251 MB,
isn't tracked either) — large generated corpora are local-only. `xmark_10.0.xml` and
all eight generated parquet files (`prepost_ids_elements.parquet`,
`prepost_ids_full.parquet`, `level_elements.parquet`, `pre_elements.parquet`,
`post_elements.parquet`, `level_full.parquet`, `pre_full.parquet`, `post_full.parquet`)
follow that convention.

## Dependencies

Install from `../requirements.txt` (needs `pandas`, `numpy`, and `pyarrow`).
