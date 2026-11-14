"""Git provenance preserves caller choices for missing repos and untracked files."""
import subprocess

import pytest

from utils.provenance import git_commit, git_identity


@pytest.fixture(autouse=True)
def isolated_git_configuration(monkeypatch, tmp_path):
    # The user's Git status/signing/hooks configuration must not affect these fixtures.
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "empty-global-config"))
    for variable in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(variable, raising=False)


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL).strip()


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / "repo"
    path.mkdir()
    git(path, "init", "--quiet")
    git(path, "config", "user.email", "test@example.invalid")
    git(path, "config", "user.name", "Test")
    (path / "tracked.txt").write_text("original\n")
    git(path, "add", "tracked.txt")
    git(path, "commit", "--quiet", "-m", "Initial fixture")
    return path


def test_missing_repository_has_fallback_or_explicitly_required_commit(tmp_path):
    assert git_commit(tmp_path) == "unknown"
    assert git_commit(tmp_path, default=None) is None
    with pytest.raises(subprocess.CalledProcessError):
        git_commit(tmp_path, required=True)
    assert git_identity(tmp_path) == {"git_commit": None, "git_dirty": None}


def test_identity_tracks_the_commit_and_tracked_edits(repo):
    commit = git(repo, "rev-parse", "HEAD")
    assert git_commit(repo, required=True) == commit
    assert git_identity(repo) == {"git_commit": commit, "git_dirty": False}
    (repo / "tracked.txt").write_text("changed\n")
    assert git_identity(repo) == {"git_commit": commit, "git_dirty": True}


def test_explicit_normal_untracked_mode_overrides_hidden_untracked_config(repo):
    git(repo, "config", "status.showUntrackedFiles", "no")
    (repo / "new-evidence.json").write_text("{}\n")
    # Reports retain Git's configured default; telemetry explicitly includes new files.
    assert git_identity(repo)["git_dirty"] is False
    assert git_identity(repo, timeout=5, untracked_files="normal")["git_dirty"] is True
    assert git_identity(repo, untracked_files="no")["git_dirty"] is False
