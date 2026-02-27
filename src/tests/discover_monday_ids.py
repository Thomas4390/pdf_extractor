"""
Discover Monday.com workspace, folder, and board IDs for provisioning setup.

Lists all workspaces, folders, sub-folders, and boards to help configure
the provisioning environment variables.

Usage:
    python -m src.tests.discover_monday_ids
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.clients.monday import MondayClient, MondayError


def discover():
    """Discover and print Monday.com workspace/folder/board structure."""
    api_key = os.environ.get("MONDAY_API_KEY")
    if not api_key:
        print("ERROR: MONDAY_API_KEY not set in environment")
        sys.exit(1)

    client = MondayClient(api_key=api_key)

    # Step 1: List workspaces
    print("=" * 60)
    print("STEP 1: Workspaces")
    print("=" * 60)

    try:
        result = client._execute_query_sync("{ workspaces { id name } }")
        workspaces = result["data"]["workspaces"]
    except MondayError as e:
        print(f"ERROR listing workspaces: {e}")
        sys.exit(1)

    if not workspaces:
        print("No workspaces found.")
        sys.exit(1)

    for ws in workspaces:
        print(f"  Workspace: {ws['name']} (ID: {ws['id']})")

    # Step 2: For each workspace, list folders
    print()
    print("=" * 60)
    print("STEP 2: Folders per workspace")
    print("=" * 60)

    conseillers_folder_id = None
    template_folder_id = None
    target_workspace_id = None

    for ws in workspaces:
        ws_id = int(ws["id"])
        print(f"\n  Workspace: {ws['name']} (ID: {ws_id})")
        print("  " + "-" * 50)

        try:
            folders = client.list_folders_sync(ws_id)
        except MondayError as e:
            print(f"    ERROR: {e}")
            continue

        if not folders:
            print("    (no folders)")
            continue

        for folder in folders:
            print(f"    Folder: {folder['name']} (ID: {folder['id']})")
            children = folder.get("children", [])
            for child in children:
                print(f"      Sub-folder: {child['name']} (ID: {child['id']})")

                # Try to identify key folders
                child_name_lower = child["name"].lower()
                if "thomas" in child_name_lower and "vaudescal" in child_name_lower:
                    template_folder_id = child["id"]
                    target_workspace_id = ws_id

            folder_name_lower = folder["name"].lower()
            if "conseiller" in folder_name_lower:
                conseillers_folder_id = folder["id"]
                target_workspace_id = ws_id

    # Step 3: If template folder found, list its boards
    print()
    print("=" * 60)
    print("STEP 3: Template boards")
    print("=" * 60)

    template_board_ids = []
    if template_folder_id:
        print(f"\n  Template folder ID: {template_folder_id}")
        try:
            boards = client.list_boards_in_folder_sync(int(template_folder_id))
            for board in boards:
                print(f"    Board: {board['name']} (ID: {board['id']})")
                template_board_ids.append(board["id"])
        except MondayError as e:
            print(f"    ERROR: {e}")
    else:
        print("  Template folder not auto-detected.")
        print("  Look for the advisor folder whose boards should be used as templates.")

    # Step 4: Summary with env vars
    print()
    print("=" * 60)
    print("STEP 4: Environment variables to configure")
    print("=" * 60)
    print()

    if target_workspace_id:
        print(f"MONDAY_WORKSPACE_ID={target_workspace_id}")
    else:
        print("MONDAY_WORKSPACE_ID=<set_manually>")

    if conseillers_folder_id:
        print(f"MONDAY_CONSEILLERS_FOLDER_ID={conseillers_folder_id}")
    else:
        print("MONDAY_CONSEILLERS_FOLDER_ID=<set_manually>")

    if template_folder_id:
        print(f"MONDAY_TEMPLATE_FOLDER_ID={template_folder_id}")
    else:
        print("MONDAY_TEMPLATE_FOLDER_ID=<set_manually>")

    if template_board_ids:
        print(f"MONDAY_TEMPLATE_BOARD_IDS={','.join(str(b) for b in template_board_ids)}")
    else:
        print("# MONDAY_TEMPLATE_BOARD_IDS=<optional, auto-discovered from folder>")

    print("# MONDAY_TEMPLATE_FIRST_NAME=Thomas  # default")
    print()


if __name__ == "__main__":
    discover()
