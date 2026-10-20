"""Fetch and verify the two pinned x86_64 wheels before entering BuildKit.

Uses only host Python's standard library; no training packages are installed
on the host. The build helper removes this temporary wheel directory on exit.
"""
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

destination = Path(sys.argv[1])
for package, version in (('av', '16.0.1'), ('bitsandbytes', '0.50.2')):
    with urllib.request.urlopen(f'https://pypi.org/pypi/{package}/{version}/json', timeout=30) as response:
        metadata = json.load(response)
    wheels = [item for item in metadata['urls']
              if 'manylinux' in item['filename'] and 'x86_64' in item['filename']
              and ('cp312-cp312' in item['filename'] or 'py3-none' in item['filename'])]
    if not wheels:
        raise RuntimeError(f'No Python 3.12 Linux x86_64 wheel for {package}=={version}')
    item = wheels[0]
    filename = item['filename']
    if Path(filename).name != filename:
        raise RuntimeError('Invalid wheel filename from package metadata')
    target = destination / filename
    if target.exists():
        with target.open('rb') as stream:
            existing = hashlib.file_digest(stream, 'sha256').hexdigest()
        if existing != item['digests']['sha256']:
            raise RuntimeError(f'SHA-256 mismatch: {filename}')
        print(f'Verified cached {package}=={version}: {filename}', flush=True)
        continue
    digest = hashlib.sha256()
    with urllib.request.urlopen(item['url'], timeout=30) as response, target.open('xb') as stream:
        while block := response.read(1024 * 1024):
            stream.write(block)
            digest.update(block)
    if digest.hexdigest() != item['digests']['sha256']:
        raise RuntimeError(f'SHA-256 mismatch: {filename}')
    print(f'Verified {package}=={version}: {filename}', flush=True)
