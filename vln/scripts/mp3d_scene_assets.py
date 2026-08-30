#!/usr/bin/env python3
"""Validate and safely restore Habitat MP3D scene assets.

This module intentionally has no Habitat or model-runtime dependency.  It
checks the binary container/layout invariants that must hold before Habitat-
Sim attempts to load a scene and aborts inside native Magnum code.
"""

import contextlib
import datetime
import fcntl
import hashlib
import json
import mmap
import os
import re
import shutil
import stat
import struct
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


DEFAULT_SCENE = "1pXnuDYAj8r"
ARCHIVE_ASSET_ID = "mp3d_habitat_archive"
KNOWN_BAD_SEMANTIC_SIZE = 115277824
KNOWN_BAD_FIRST_FACE_SIZE = 64
MAX_PLY_HEADER_SIZE = 1024 * 1024

REQUIRED_FILE_TEMPLATES = (
    "{scene}.glb",
    "{scene}.navmesh",
    "{scene}.house",
    "{scene}_semantic.ply",
)

PLY_TYPES = {
    "char": ("b", 1, True),
    "int8": ("b", 1, True),
    "uchar": ("B", 1, True),
    "uint8": ("B", 1, True),
    "short": ("h", 2, True),
    "int16": ("h", 2, True),
    "ushort": ("H", 2, True),
    "uint16": ("H", 2, True),
    "int": ("i", 4, True),
    "int32": ("i", 4, True),
    "uint": ("I", 4, True),
    "uint32": ("I", 4, True),
    "float": ("f", 4, False),
    "float32": ("f", 4, False),
    "double": ("d", 8, False),
    "float64": ("d", 8, False),
}

SCENE_ID_PATTERN = re.compile(r"^[A-Za-z0-9]{11}$")


class SceneAssetError(RuntimeError):
    """Raised when a scene asset or its provenance is invalid."""


def sha256_file(path, chunk_size=8 * 1024 * 1024, progress=None):
    digest = hashlib.sha256()
    total = os.path.getsize(path)
    completed = 0
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
            completed += len(chunk)
            if progress is not None:
                progress(completed, total)
    return digest.hexdigest()


def validate_scene_id(scene):
    if not isinstance(scene, str) or SCENE_ID_PATTERN.fullmatch(scene) is None:
        raise SceneAssetError(
            "invalid MP3D scene identifier {!r}; expected 11 ASCII letters/digits"
            .format(scene)
        )
    return scene


def required_filenames(scene):
    scene = validate_scene_id(scene)
    return tuple(template.format(scene=scene) for template in REQUIRED_FILE_TEMPLATES)


def _require_regular_nonempty(path, label):
    if not path.is_file():
        raise SceneAssetError("missing {}: {}".format(label, path))
    size = path.stat().st_size
    if size <= 0:
        raise SceneAssetError("empty {}: {}".format(label, path))
    return size


def validate_glb(path):
    """Validate GLB magic, version, declared length, and chunk boundaries."""
    path = Path(path)
    size = _require_regular_nonempty(path, "GLB")
    if size < 20:
        raise SceneAssetError("GLB is shorter than its header and first chunk: {}".format(path))

    with path.open("rb") as stream:
        header = stream.read(12)
        magic, version, declared_length = struct.unpack("<4sII", header)
        if magic != b"glTF":
            raise SceneAssetError(
                "invalid GLB magic in {}: {!r}".format(path, magic)
            )
        if version != 2:
            raise SceneAssetError(
                "unsupported GLB version in {}: {}".format(path, version)
            )
        if declared_length != size:
            raise SceneAssetError(
                "GLB declared length mismatch in {}: header={}, file={}".format(
                    path, declared_length, size
                )
            )

        offset = 12
        chunk_count = 0
        first_chunk_type = None
        while offset < size:
            chunk_header = stream.read(8)
            if len(chunk_header) != 8:
                raise SceneAssetError(
                    "truncated GLB chunk header at byte {} in {}".format(offset, path)
                )
            chunk_length, chunk_type = struct.unpack("<II", chunk_header)
            if chunk_length % 4 != 0:
                raise SceneAssetError(
                    "unaligned GLB chunk length {} at byte {} in {}".format(
                        chunk_length, offset, path
                    )
                )
            if first_chunk_type is None:
                first_chunk_type = chunk_type
            offset += 8 + chunk_length
            if offset > size:
                raise SceneAssetError(
                    "GLB chunk overruns declared length in {}".format(path)
                )
            stream.seek(chunk_length, os.SEEK_CUR)
            chunk_count += 1

    if chunk_count == 0 or first_chunk_type != 0x4E4F534A:
        raise SceneAssetError("GLB first chunk is not JSON in {}".format(path))
    if offset != size:
        raise SceneAssetError("GLB chunk layout does not consume {}".format(path))

    return {
        "path": str(path),
        "size": size,
        "version": version,
        "chunks": chunk_count,
    }


def _parse_ply_header(stream, path):
    consumed = 0
    lines = []
    while consumed <= MAX_PLY_HEADER_SIZE:
        line = stream.readline(MAX_PLY_HEADER_SIZE + 1 - consumed)
        if not line:
            raise SceneAssetError("PLY header has no end_header marker: {}".format(path))
        consumed += len(line)
        if consumed > MAX_PLY_HEADER_SIZE:
            raise SceneAssetError("PLY header is larger than {} bytes: {}".format(
                MAX_PLY_HEADER_SIZE, path
            ))
        try:
            decoded = line.rstrip(b"\r\n").decode("ascii")
        except UnicodeDecodeError as error:
            raise SceneAssetError("PLY header is not ASCII in {}: {}".format(path, error))
        lines.append(decoded)
        if decoded == "end_header":
            break

    if not lines or lines[0] != "ply":
        raise SceneAssetError("invalid PLY magic in {}".format(path))

    file_format = None
    elements = []
    current = None
    for line in lines[1:]:
        fields = line.split()
        if not fields or fields[0] in ("comment", "obj_info", "end_header"):
            continue
        directive = fields[0]
        if directive == "format":
            if len(fields) != 3:
                raise SceneAssetError("malformed PLY format line in {}".format(path))
            file_format = fields[1]
        elif directive == "element":
            if len(fields) != 3:
                raise SceneAssetError("malformed PLY element line in {}".format(path))
            try:
                count = int(fields[2])
            except ValueError:
                raise SceneAssetError("invalid PLY element count in {}".format(path))
            if count < 0:
                raise SceneAssetError("negative PLY element count in {}".format(path))
            current = {"name": fields[1], "count": count, "properties": []}
            elements.append(current)
        elif directive == "property":
            if current is None:
                raise SceneAssetError("PLY property precedes element in {}".format(path))
            if len(fields) == 3:
                type_name, name = fields[1], fields[2]
                if type_name not in PLY_TYPES:
                    raise SceneAssetError(
                        "unsupported PLY scalar type {} in {}".format(type_name, path)
                    )
                current["properties"].append(
                    {"kind": "scalar", "type": type_name, "name": name}
                )
            elif len(fields) == 5 and fields[1] == "list":
                count_type, item_type, name = fields[2], fields[3], fields[4]
                if count_type not in PLY_TYPES or item_type not in PLY_TYPES:
                    raise SceneAssetError("unsupported PLY list type in {}".format(path))
                if not PLY_TYPES[count_type][2]:
                    raise SceneAssetError(
                        "PLY list count type must be integral in {}".format(path)
                    )
                current["properties"].append(
                    {
                        "kind": "list",
                        "count_type": count_type,
                        "item_type": item_type,
                        "name": name,
                    }
                )
            else:
                raise SceneAssetError("malformed PLY property line in {}".format(path))
        else:
            raise SceneAssetError(
                "unsupported PLY header directive {} in {}".format(directive, path)
            )

    if file_format != "binary_little_endian":
        raise SceneAssetError(
            "semantic PLY must be binary_little_endian, got {} in {}".format(
                file_format, path
            )
        )
    names = [element["name"] for element in elements]
    if names.count("vertex") != 1 or names.count("face") != 1:
        raise SceneAssetError(
            "semantic PLY must declare exactly one vertex and one face element: {}".format(
                path
            )
        )
    return consumed, elements


def _read_integer(buffer, offset, type_name, path):
    format_code, width, integral = PLY_TYPES[type_name]
    if not integral:
        raise SceneAssetError("non-integral PLY count type in {}".format(path))
    if offset + width > len(buffer):
        raise SceneAssetError(
            "PLY payload is truncated at byte {} in {}".format(offset, path)
        )
    return struct.unpack_from("<" + format_code, buffer, offset)[0], width


def validate_semantic_ply(path, scene=None):
    """Validate the declared binary PLY element layout through exact EOF."""
    path = Path(path)
    if scene is not None:
        validate_scene_id(scene)
    size = _require_regular_nonempty(path, "semantic PLY")
    with path.open("rb") as stream:
        header_size, elements = _parse_ply_header(stream, path)
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as payload:
            offset = header_size
            vertex_count = next(
                element["count"] for element in elements if element["name"] == "vertex"
            )
            face_count = next(
                element["count"] for element in elements if element["name"] == "face"
            )
            face_offset = None
            first_face_size = None

            for element in elements:
                properties = element["properties"]
                if not properties and element["count"]:
                    raise SceneAssetError(
                        "PLY element {} has no properties in {}".format(
                            element["name"], path
                        )
                    )

                if all(prop["kind"] == "scalar" for prop in properties):
                    record_size = sum(PLY_TYPES[prop["type"]][1] for prop in properties)
                    end = offset + element["count"] * record_size
                    if end > size:
                        raise SceneAssetError(
                            "PLY {} payload is truncated: declared end {}, file size {} ({})"
                            .format(element["name"], end, size, path)
                        )
                    if element["name"] == "face":
                        raise SceneAssetError(
                            "PLY face element has no vertex-index list in {}".format(path)
                        )
                    offset = end
                    continue

                vertex_index_properties = [
                    prop
                    for prop in properties
                    if prop["kind"] == "list"
                    and prop["name"] in ("vertex_indices", "vertex_index")
                ]
                if element["name"] == "face" and len(vertex_index_properties) != 1:
                    raise SceneAssetError(
                        "PLY face element needs one vertex-index list in {}".format(path)
                    )

                for record_index in range(element["count"]):
                    if element["name"] == "face" and record_index == 0:
                        face_offset = offset
                    for prop in properties:
                        if prop["kind"] == "scalar":
                            width = PLY_TYPES[prop["type"]][1]
                            if offset + width > size:
                                raise SceneAssetError(
                                    "PLY payload is truncated at byte {} in {}".format(
                                        offset, path
                                    )
                                )
                            offset += width
                            continue

                        count, count_width = _read_integer(
                            payload, offset, prop["count_type"], path
                        )
                        if count < 0:
                            raise SceneAssetError(
                                "negative PLY list length at byte {} in {}".format(
                                    offset, path
                                )
                            )
                        if (
                            element["name"] == "face"
                            and prop["name"] in ("vertex_indices", "vertex_index")
                        ):
                            if record_index == 0:
                                first_face_size = count
                            if count not in (3, 4):
                                raw_first_byte = (
                                    payload[face_offset]
                                    if face_offset is not None
                                    else None
                                )
                                known_scene = scene or path.name.replace("_semantic.ply", "")
                                if (
                                    known_scene == DEFAULT_SCENE
                                    and size == KNOWN_BAD_SEMANTIC_SIZE
                                    and raw_first_byte == KNOWN_BAD_FIRST_FACE_SIZE
                                ):
                                    raise SceneAssetError(
                                        "known corrupt {} semantic PLY signature: size={}, "
                                        "first face byte={} (header-declared face offset {}); "
                                        "restore the scene from the verified mp3d_habitat archive"
                                        .format(
                                            DEFAULT_SCENE,
                                            size,
                                            raw_first_byte,
                                            face_offset,
                                        )
                                    )
                                raise SceneAssetError(
                                    "unsupported PLY face size {} at face {} (byte {}) in {}"
                                    .format(count, record_index, offset, path)
                                )
                        item_width = PLY_TYPES[prop["item_type"]][1]
                        end = offset + count_width + count * item_width
                        if end > size:
                            raise SceneAssetError(
                                "PLY list at byte {} overruns file size {} in {}".format(
                                    offset, size, path
                                )
                            )
                        offset = end

            if offset != size:
                raise SceneAssetError(
                    "PLY header-declared layout ends at byte {}, but file size is {} "
                    "(trailing/padded bytes) in {}".format(offset, size, path)
                )

    return {
        "path": str(path),
        "size": size,
        "header_size": header_size,
        "vertices": vertex_count,
        "faces": face_count,
        "first_face_offset": face_offset,
        "first_face_size": first_face_size,
    }


def validate_scene_directory(scene_directory, scene=None):
    scene_directory = Path(scene_directory)
    scene = scene or scene_directory.name
    validate_scene_id(scene)
    if not scene_directory.is_dir():
        raise SceneAssetError("missing scene directory: {}".format(scene_directory))

    paths = {
        "glb": scene_directory / (scene + ".glb"),
        "navmesh": scene_directory / (scene + ".navmesh"),
        "house": scene_directory / (scene + ".house"),
        "semantic_ply": scene_directory / (scene + "_semantic.ply"),
    }
    report = {
        "scene": scene,
        "directory": str(scene_directory),
        "glb": validate_glb(paths["glb"]),
        "navmesh": {
            "path": str(paths["navmesh"]),
            "size": _require_regular_nonempty(paths["navmesh"], "navmesh"),
        },
        "house": {
            "path": str(paths["house"]),
            "size": _require_regular_nonempty(paths["house"], "house"),
        },
        "semantic_ply": validate_semantic_ply(paths["semantic_ply"], scene=scene),
    }
    return report


def load_archive_spec(repo_root, manifest_path=None, archive_path=None):
    repo_root = Path(repo_root).resolve()
    if manifest_path is None:
        manifest_path = repo_root / "vln/manifests/assets/eval_assets.json"
    else:
        manifest_path = Path(manifest_path)
    try:
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
    except (OSError, ValueError) as error:
        raise SceneAssetError(
            "cannot read asset manifest {}: {}".format(manifest_path, error)
        )

    matches = [
        asset
        for asset in manifest.get("assets", [])
        if asset.get("id") == ARCHIVE_ASSET_ID
    ]
    if len(matches) != 1:
        raise SceneAssetError(
            "manifest must contain exactly one {} asset".format(ARCHIVE_ASSET_ID)
        )
    asset = matches[0]
    if asset.get("source") != "mp3d_habitat":
        raise SceneAssetError(
            "{} must reference the mp3d_habitat source".format(ARCHIVE_ASSET_ID)
        )
    expected_size = asset.get("size")
    expected_sha256 = asset.get("sha256")
    if type(expected_size) is not int or expected_size <= 0:
        raise SceneAssetError("archive manifest size is invalid")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise SceneAssetError("archive manifest SHA256 is invalid")

    if archive_path is None:
        declared_path = asset.get("path")
        if not isinstance(declared_path, str) or not declared_path:
            raise SceneAssetError("archive manifest path is invalid")
        archive_path = Path(declared_path)
        if not archive_path.is_absolute():
            archive_path = repo_root / archive_path
    else:
        archive_path = Path(archive_path)
    return {
        "manifest_path": str(manifest_path),
        "archive_path": str(archive_path),
        "expected_size": expected_size,
        "expected_sha256": expected_sha256,
        "source": asset.get("source"),
    }


def verify_archive(spec, progress=None):
    archive_path = Path(spec["archive_path"])
    if not archive_path.is_file():
        raise SceneAssetError("missing MP3D archive: {}".format(archive_path))
    actual_size = archive_path.stat().st_size
    if actual_size != spec["expected_size"]:
        raise SceneAssetError(
            "MP3D archive size mismatch: expected {}, got {} ({})".format(
                spec["expected_size"], actual_size, archive_path
            )
        )
    actual_sha256 = sha256_file(archive_path, progress=progress)
    if actual_sha256 != spec["expected_sha256"]:
        raise SceneAssetError(
            "MP3D archive SHA256 mismatch: expected {}, got {} ({})".format(
                spec["expected_sha256"], actual_sha256, archive_path
            )
        )
    if not zipfile.is_zipfile(str(archive_path)):
        raise SceneAssetError("verified MP3D archive is not a ZIP: {}".format(archive_path))
    verified = dict(spec)
    verified["actual_size"] = actual_size
    verified["actual_sha256"] = actual_sha256
    return verified


def _zip_member_map(archive, scene):
    expected_names = set(required_filenames(scene))
    matches = {name: [] for name in expected_names}
    for info in archive.infolist():
        name = info.filename
        if "\\" in name:
            continue
        parts = PurePosixPath(name).parts
        if len(parts) < 2 or parts[-2] != scene or parts[-1] not in expected_names:
            continue
        if info.is_dir():
            continue
        mode = (info.external_attr >> 16) & 0xFFFF
        if mode and stat.S_ISLNK(mode):
            raise SceneAssetError("archive scene member is a symlink: {}".format(name))
        matches[parts[-1]].append(info)

    errors = []
    selected = {}
    for filename in sorted(expected_names):
        candidates = matches[filename]
        if not candidates:
            errors.append("missing {}".format(filename))
        elif len(candidates) > 1:
            errors.append(
                "duplicate {} members: {}".format(
                    filename, [candidate.filename for candidate in candidates]
                )
            )
        else:
            selected[filename] = candidates[0]
    if errors:
        raise SceneAssetError(
            "archive does not contain one unambiguous {} scene: {}".format(
                scene, "; ".join(errors)
            )
        )
    return selected


def _extract_scene(archive_path, scene, destination):
    destination = Path(destination)
    destination.mkdir(mode=0o755)
    with zipfile.ZipFile(str(archive_path), "r") as archive:
        members = _zip_member_map(archive, scene)
        for filename in required_filenames(scene):
            target = destination / filename
            try:
                with archive.open(members[filename], "r") as source, target.open("xb") as sink:
                    shutil.copyfileobj(source, sink, length=8 * 1024 * 1024)
                    sink.flush()
                    os.fsync(sink.fileno())
            except (OSError, RuntimeError, zipfile.BadZipFile) as error:
                raise SceneAssetError(
                    "cannot extract {} from {}: {}".format(
                        members[filename].filename, archive_path, error
                    )
                )


def _next_backup_path(scene_root, scene):
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = scene_root / (scene + ".backup-" + timestamp)
    candidate = base
    counter = 1
    while os.path.lexists(str(candidate)):
        candidate = Path(str(base) + ".{}".format(counter))
        counter += 1
    return candidate


def _fsync_directory(path):
    try:
        descriptor = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


@contextlib.contextmanager
def _repair_lock(scene_root):
    lock_path = Path(scene_root) / ".mp3d-scene-repair.lock"
    with lock_path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise SceneAssetError(
                "another MP3D scene repair holds {}: {}".format(lock_path, error)
            )
        yield


def repair_scene(
    repo_root,
    scene=DEFAULT_SCENE,
    scene_root=None,
    manifest_path=None,
    archive_path=None,
    dry_run=False,
    force=False,
    progress=None,
):
    """Restore one scene from a fully manifest-verified official archive."""
    repo_root = Path(repo_root).resolve()
    validate_scene_id(scene)
    if scene_root is None:
        scene_root = repo_root / "vln/data/scene_datasets/mp3d"
    else:
        scene_root = Path(scene_root)
    if not scene_root.is_dir():
        raise SceneAssetError("missing MP3D scene root: {}".format(scene_root))

    target = scene_root / scene
    if target.is_symlink():
        raise SceneAssetError("refusing to replace symlinked scene directory: {}".format(target))
    if target.exists() and not target.is_dir():
        raise SceneAssetError("scene target is not a directory: {}".format(target))
    if target.is_dir() and not force:
        try:
            validate_scene_directory(target, scene=scene)
        except SceneAssetError:
            pass
        else:
            raise SceneAssetError(
                "scene already passes integrity checks; use --force to replace it: {}"
                .format(target)
            )

    archive_spec = load_archive_spec(
        repo_root, manifest_path=manifest_path, archive_path=archive_path
    )
    verified_archive = verify_archive(archive_spec, progress=progress)

    with _repair_lock(scene_root):
        staging_root = Path(
            tempfile.mkdtemp(prefix=".{}.repair-".format(scene), dir=str(scene_root))
        )
        staged_scene = staging_root / scene
        try:
            _extract_scene(verified_archive["archive_path"], scene, staged_scene)
            staged_report = validate_scene_directory(staged_scene, scene=scene)
            if dry_run:
                return {
                    "scene": scene,
                    "dry_run": True,
                    "archive": verified_archive,
                    "staged_validation": staged_report,
                    "target": str(target),
                    "backup": None,
                }

            backup = None
            if os.path.lexists(str(target)):
                backup = _next_backup_path(scene_root, scene)
                os.replace(str(target), str(backup))
                _fsync_directory(scene_root)
            try:
                os.replace(str(staged_scene), str(target))
                _fsync_directory(scene_root)
            except BaseException:
                if backup is not None and not os.path.lexists(str(target)):
                    os.replace(str(backup), str(target))
                    _fsync_directory(scene_root)
                raise

            try:
                installed_report = validate_scene_directory(target, scene=scene)
            except BaseException as validation_error:
                try:
                    if os.path.lexists(str(target)):
                        os.replace(str(target), str(staged_scene))
                    if backup is not None:
                        os.replace(str(backup), str(target))
                    _fsync_directory(scene_root)
                except BaseException as rollback_error:
                    raise SceneAssetError(
                        "installed scene validation failed ({}) and rollback failed ({})"
                        .format(validation_error, rollback_error)
                    ) from validation_error
                raise
            return {
                "scene": scene,
                "dry_run": False,
                "archive": verified_archive,
                "staged_validation": staged_report,
                "installed_validation": installed_report,
                "target": str(target),
                "backup": None if backup is None else str(backup),
            }
        finally:
            shutil.rmtree(str(staging_root), ignore_errors=True)
