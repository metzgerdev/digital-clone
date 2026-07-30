"""Build an author's style profile from their sent email (Agent 2: Profiler).

Scans a maildir tree, ranks employees by sent volume, selects one with a
substantial history (>=200), and builds a StyleProfile from their sent mail.
Writes to ../profiles/profile_<name>.json.

    python agents/build_profile.py <maildir_root>
    TARGET=kaminski-v python agents/build_profile.py <maildir_root>
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # locate style_extractor
from style_extractor import StyleProfile, parse_enron_message

SENT_FOLDERS = ("_sent_mail", "sent", "sent_items")
MIN_SENT = 200
MAX_EMAILS = 3000  # safety cap on runtime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILES = os.path.join(ROOT, "profiles")


def sent_files_by_user(maildir):
    counts = {}
    for user in sorted(os.listdir(maildir)):
        udir = os.path.join(maildir, user)
        if not os.path.isdir(udir):
            continue
        files = []
        for sf in SENT_FOLDERS:
            p = os.path.join(udir, sf)
            if os.path.isdir(p):
                files += [
                    os.path.join(p, f)
                    for f in sorted(os.listdir(p))
                    if os.path.isfile(os.path.join(p, f))
                ]
        if files:
            counts[user] = files
    return counts


def load_bodies(files):
    bodies = []
    for fp in files[:MAX_EMAILS]:
        try:
            with open(fp, "r", encoding="latin-1") as fh:
                bodies.append(parse_enron_message(fh.read()))
        except Exception:
            continue
    return bodies


def main():
    maildir = sys.argv[1] if len(sys.argv) > 1 else "maildir"
    counts = sent_files_by_user(maildir)
    if not counts:
        sys.exit(f"no sent folders found under {maildir}")

    ranked = sorted(counts.items(), key=lambda kv: len(kv[1]), reverse=True)
    print("Top senders by sent-folder message count:")
    for user, files in ranked[:15]:
        print(f"  {user:<22} {len(files):>5}")

    target = os.environ.get("TARGET")
    if not target:
        target = next((u for u, f in ranked if len(f) >= MIN_SENT), ranked[0][0])
    files = counts[target]
    print(f"\nSelected employee: {target}  ({len(files)} sent emails, using up to {MAX_EMAILS})")

    bodies = load_bodies(files)
    profile = StyleProfile().build(bodies)
    print("\n" + json.dumps(profile, indent=2))

    os.makedirs(PROFILES, exist_ok=True)
    out_path = os.path.join(PROFILES, f"profile_{target.replace('/', '_')}.json")
    with open(out_path, "w") as fh:
        json.dump(
            {
                "employee": target,
                "n_sent_files": len(files),
                "n_used": min(len(files), MAX_EMAILS),
                "profile": profile,
            },
            fh,
            indent=2,
        )
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
