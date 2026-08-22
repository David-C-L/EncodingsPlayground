#!/usr/bin/env python3
"""Generate pre/post/level tree-position ids from an XMark XML corpus.

Reads an XMark-generated XML file (see README.md for how to produce one with
xmlgen.Linux) and writes, for the chosen node model (--mode):

  - prepost_ids_<mode>.parquet  column "prepost_id", int64: level/pre/post
    bit-packed into one 64-bit key (see README.md for the bit layout).
  - level_<mode>.parquet        column "level", int32
  - pre_<mode>.parquet          column "pre",   int32
  - post_<mode>.parquet         column "post",  int32

Traversal is iterative (xml.etree.ElementTree.iterparse, driven by the C expat
parser) so it never recurses in Python and never loads the file as a DOM.
"""

import argparse
import array
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

# Bit layout for the packed 64-bit id: level | pre | post, MSB to LSB.
# Sized to the real xmark_10.0.xml corpus (16.7M elements, 32.3M nodes under
# "full", max level 12) with headroom; see README.md for the full rationale.
LEVEL_BITS = 8
PRE_BITS = 28
POST_BITS = 28
assert LEVEL_BITS + PRE_BITS + POST_BITS == 64

LEVEL_MAX = (1 << LEVEL_BITS) - 1
PRE_MAX = (1 << PRE_BITS) - 1
POST_MAX = (1 << POST_BITS) - 1


def pack_ids(level: np.ndarray, pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    """Bit-pack parallel level/pre/post arrays into one uint64 array."""
    for name, values, limit in (("level", level, LEVEL_MAX),
                                 ("pre", pre, PRE_MAX),
                                 ("post", post, POST_MAX)):
        overflow = int(values.max(initial=0))
        if overflow > limit:
            raise ValueError(
                f"{name} value {overflow} exceeds its {limit.bit_length()}-bit "
                f"budget ({limit}); widen the corresponding *_BITS constant.")

    level64 = level.astype(np.uint64)
    pre64 = pre.astype(np.uint64)
    post64 = post.astype(np.uint64)
    return (level64 << np.uint64(PRE_BITS + POST_BITS)) | (pre64 << np.uint64(POST_BITS)) | post64


class _Frame:
    """Per-open-element traversal state, kept on an explicit stack."""

    __slots__ = ("idx", "depth", "elem", "text_flushed", "pending_tail_elem")

    def __init__(self, idx, depth, elem):
        self.idx = idx
        self.depth = depth
        self.elem = elem
        self.text_flushed = False
        self.pending_tail_elem = None


def _walk(xml_path: Path, full: bool):
    """Iterative DFS over xml_path. Yields nothing; fills and returns three
    array.array('I' or 'B', ...) buffers of level/pre/post, index-aligned by
    the node's `pre` value (buffers are appended to in document/pre order, so
    an element's post is back-filled by index once its `end` event, or a leaf's
    post is filled immediately since leaves have no children to visit first).
    """
    levels = array.array("I")
    pres = array.array("I")
    posts = array.array("I")
    post_counter = 0

    def emit_leaf(depth: int) -> None:
        nonlocal post_counter
        idx = len(levels)
        levels.append(depth)
        pres.append(idx)
        posts.append(post_counter)
        post_counter += 1

    def emit_element_start(depth: int) -> int:
        idx = len(levels)
        levels.append(depth)
        pres.append(idx)
        posts.append(0)  # placeholder, back-filled at the matching `end`
        return idx

    def flush_leading_text(frame: "_Frame") -> None:
        if not frame.text_flushed:
            if frame.elem.text and frame.elem.text.strip():
                emit_leaf(frame.depth + 1)
            frame.text_flushed = True

    def flush_pending_tail(frame: "_Frame") -> None:
        pending = frame.pending_tail_elem
        if pending is not None:
            if pending.tail and pending.tail.strip():
                emit_leaf(frame.depth + 1)
            pending.clear()
            frame.pending_tail_elem = None

    depth = 0
    stack: list[_Frame] = []
    for event, elem in ET.iterparse(str(xml_path), events=("start", "end")):
        if event == "start":
            if stack:
                parent = stack[-1]
                if full:
                    flush_leading_text(parent)
                    flush_pending_tail(parent)
            idx = emit_element_start(depth)
            if full:
                for _ in elem.attrib:
                    emit_leaf(depth + 1)
            stack.append(_Frame(idx, depth, elem))
            depth += 1
        else:  # "end"
            depth -= 1
            frame = stack.pop()
            if full:
                flush_leading_text(frame)
                flush_pending_tail(frame)
            posts[frame.idx] = post_counter
            post_counter += 1
            if stack:
                if full:
                    stack[-1].pending_tail_elem = elem
            else:
                elem.clear()

    return levels, pres, posts


def generate(xml_path: Path, mode: str):
    levels, pres, posts = _walk(xml_path, full=(mode == "full"))
    level = np.frombuffer(levels, dtype=np.uint32)
    pre = np.frombuffer(pres, dtype=np.uint32)
    post = np.frombuffer(posts, dtype=np.uint32)
    return level, pre, post


def write_columns(outdir: Path, mode: str, level: np.ndarray, pre: np.ndarray,
                   post: np.ndarray) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    packed = pack_ids(level, pre, post).view(np.int64)
    pd.DataFrame({"prepost_id": packed}).to_parquet(
        outdir / f"prepost_ids_{mode}.parquet", index=False)

    pd.DataFrame({"level": level.view(np.int32)}).to_parquet(
        outdir / f"level_{mode}.parquet", index=False)
    pd.DataFrame({"pre": pre.view(np.int32)}).to_parquet(
        outdir / f"pre_{mode}.parquet", index=False)
    pd.DataFrame({"post": post.view(np.int32)}).to_parquet(
        outdir / f"post_{mode}.parquet", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate pre/post/level ids from an XMark XML corpus.")
    parser.add_argument("--xml", type=Path, default=Path(__file__).parent / "xmark_10.0.xml")
    parser.add_argument("--mode", choices=("elements", "full"), required=True)
    parser.add_argument("--outdir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()

    level, pre, post = generate(args.xml, args.mode)
    print(f"mode={args.mode} nodes={len(level)} max_level={int(level.max())} "
          f"max_pre={int(pre.max())} max_post={int(post.max())}")
    write_columns(args.outdir, args.mode, level, pre, post)


if __name__ == "__main__":
    main()
