import os
import json
import hashlib
import argparse
import fnmatch
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

CACHE_FILE = ".drive_sync_state.json"
# DRIVE_FOLDER_ID = ""
IGNORE_FILE = ".driveignore"


def load_driveignore():
    ignore_patterns = set()
    if os.path.exists(IGNORE_FILE):
        with open(IGNORE_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    ignore_patterns.add(line)
    return ignore_patterns


def should_ignore(rel_path, ignore_patterns):
    for pattern in ignore_patterns:
        if pattern.endswith('/'):
            if rel_path.startswith(pattern) or f"/{pattern}" in rel_path:
                return True
        if fnmatch.fnmatch(rel_path, pattern):
            return True
    return False


def hash_file(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()


def scan_files(root_dir, ignore_patterns):
    file_hashes = {}
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, root_dir).replace(os.sep, '/')
            if should_ignore(rel_path, ignore_patterns):
                continue
            file_hashes[rel_path] = hash_file(full_path)
    return file_hashes


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_cache(file_hashes):
    with open(CACHE_FILE, 'w') as f:
        json.dump(file_hashes, f, indent=2)


def get_changed_files(current, cached):
    return [f for f in current if f not in cached or current[f] != cached[f]]


def show_status(changed):
    if not changed:
        print("No files to upload. All files are up to date.")
    else:
        print("Changed files:")
        for f in changed:
            print(f" - {f}")


def ensure_drive_path(drive, full_path):
    folders = full_path.strip('/').split('/')[:-1]
    parent_id = DRIVE_FOLDER_ID
    for folder in folders:
        query = f"title = '{folder}' and mimeType = 'application/vnd.google-apps.folder' and '{parent_id}' in parents and trashed=false"
        file_list = drive.ListFile({'q': query}).GetList()
        if file_list:
            parent_id = file_list[0]['id']
        else:
            folder_metadata = {
                'title': folder,
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [{'id': parent_id}]
            }
            folder_file = drive.CreateFile(folder_metadata)
            folder_file.Upload()
            parent_id = folder_file['id']
    return parent_id


def upload_files(file_list, root_dir, drive_folder_id):
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)

    for rel_path in file_list:
        full_path = os.path.join(root_dir, rel_path)
        rel_path_drive = rel_path.replace(os.sep, '/')

        parent_id = ensure_drive_path(drive, rel_path_drive)

        # 중복 파일 확인 후 대체
        basename = os.path.basename(rel_path_drive)
        query = f"title = '{basename}' and '{parent_id}' in parents and trashed=false"
        file_list = drive.ListFile({'q': query}).GetList()
        for existing in file_list:
            existing.Delete()

        file = drive.CreateFile({
            'title': basename,
            'parents': [{'id': parent_id}]
        })
        file.SetContentFile(full_path)
        file.Upload()
        print(f"Uploaded: {rel_path_drive}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Selective Google Drive Sync")
    parser.add_argument("command", choices=["status", "commit"], help="Command to run")
    parser.add_argument("message", nargs="?", help="Commit message")
    parser.add_argument("--root", default="./", help="Project root directory")
    args = parser.parse_args()

    ignore_patterns = load_driveignore()
    current_state = scan_files(args.root, ignore_patterns)
    cached_state = load_cache()
    changed_files = get_changed_files(current_state, cached_state)

    if args.command == "status":
        show_status(changed_files)

    elif args.command == "commit":
        if not changed_files:
            print("No changes to commit.")
        else:
            print(f"Committing {len(changed_files)} files...")
            upload_files(changed_files, args.root, DRIVE_FOLDER_ID)
            save_cache(current_state)
            if args.message:
                print(f"Commit message: {args.message}")