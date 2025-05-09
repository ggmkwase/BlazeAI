# updater.py
import os
import sys
import shutil
import subprocess

REPO_URL = "https://github.com/ggmkwase/BlazeAI.git"
UPDATE_FOLDER = "blaze_update_temp"

def update_blaze():
    try:
        # Step 1: Clone latest version
        if os.path.exists(UPDATE_FOLDER):
            shutil.rmtree(UPDATE_FOLDER)
        subprocess.run(["git", "clone", REPO_URL, UPDATE_FOLDER], check=True)

        # Step 2: Replace files in current directory
        for filename in os.listdir(UPDATE_FOLDER):
            src = os.path.join(UPDATE_FOLDER, filename)
            dst = os.path.join(os.getcwd(), filename)
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

        shutil.rmtree(UPDATE_FOLDER)
        print("✅ Blaze updated. Restarting...")

        # Step 3: Restart script
        os.execv(sys.executable, ['python'] + sys.argv)

    except Exception as e:
        print("❌ Update failed:", str(e))
        return f"❌ Update failed: {e}"
