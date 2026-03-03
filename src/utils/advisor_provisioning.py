"""
Advisor Board Provisioning for Monday.com.

Automatically creates Monday.com infrastructure when a new advisor is added:
- Dedicated folder under "Conseillers"
- Duplicated template boards (with structure: columns, settings, automations)
- Clean groups (delete old, create current month)
- Email invitation to the advisor

Also handles monthly group provisioning:
- Auto-create next month's group on all advisor boards at startup
- Persist last update month in Google Sheets "Settings" worksheet
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from src.clients.monday import MondayClient, MondayError
from src.utils.aggregator import MONTHS_FR

logger = logging.getLogger(__name__)

# Rate limit delay between board duplications (Monday.com: 40/min)
DUPLICATION_DELAY = 2.0


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ProvisioningConfig:
    """Configuration for advisor board provisioning."""
    workspace_id: int
    conseillers_folder_id: int
    template_folder_id: int
    template_board_ids: Optional[list[int]] = None  # If None, auto-discover from folder
    template_first_name: str = "Thomas"  # First name used in template board names


@dataclass
class ProvisioningStep:
    """A single step in the provisioning process."""
    name: str
    status: str = "pending"  # pending, running, success, error
    message: str = ""


@dataclass
class ProvisioningResult:
    """Result of the provisioning process."""
    success: bool = False
    advisor_folder_id: Optional[str] = None
    created_board_ids: list[str] = field(default_factory=list)
    invited_user_id: Optional[int] = None
    steps: list[ProvisioningStep] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# =============================================================================
# HELPERS
# =============================================================================

def rename_board_for_advisor(
    original_name: str,
    advisor_first_name: str,
    template_first_name: str = "Thomas",
) -> str:
    """Rename a board by replacing the template first name with the advisor's.

    Examples:
        >>> rename_board_for_advisor("Thomas - Ventes", "Jean")
        'Jean - Ventes'
        >>> rename_board_for_advisor("Thomas - Paiement Historique", "Marie")
        'Marie - Paiement Historique'
        >>> rename_board_for_advisor("Some Board", "Jean")
        'Jean - Some Board'
    """
    if " - " in original_name:
        suffix = original_name.split(" - ", 1)[1]
        return f"{advisor_first_name} - {suffix}"
    # If no " - " separator, prefix the advisor name
    return f"{advisor_first_name} - {original_name}"


def get_current_month_group_name() -> str:
    """Get the group name for the current month in French.

    Returns:
        e.g., "Février 2026"
    """
    today = date.today()
    return f"{MONTHS_FR[today.month]} {today.year}"


def load_provisioning_config() -> Optional[ProvisioningConfig]:
    """Load provisioning configuration from secrets/environment.

    Expected variables:
        MONDAY_WORKSPACE_ID: Workspace ID
        MONDAY_CONSEILLERS_FOLDER_ID: Parent folder for advisor folders
        MONDAY_TEMPLATE_FOLDER_ID: Folder containing template boards
        MONDAY_TEMPLATE_BOARD_IDS: Comma-separated board IDs (optional)
        MONDAY_TEMPLATE_FIRST_NAME: First name used in template boards (optional)

    Returns:
        ProvisioningConfig if all required values are present, None otherwise
    """
    from src.app.state import get_secret

    workspace_id = get_secret("MONDAY_WORKSPACE_ID")
    conseillers_folder_id = get_secret("MONDAY_CONSEILLERS_FOLDER_ID")
    template_folder_id = get_secret("MONDAY_TEMPLATE_FOLDER_ID")

    if not all([workspace_id, conseillers_folder_id, template_folder_id]):
        return None

    try:
        config = ProvisioningConfig(
            workspace_id=int(workspace_id),
            conseillers_folder_id=int(conseillers_folder_id),
            template_folder_id=int(template_folder_id),
        )
    except (ValueError, TypeError):
        return None

    # Optional: explicit board IDs
    template_board_ids_str = get_secret("MONDAY_TEMPLATE_BOARD_IDS")
    if template_board_ids_str:
        try:
            config.template_board_ids = [
                int(bid.strip()) for bid in template_board_ids_str.split(",") if bid.strip()
            ]
        except ValueError:
            pass

    # Optional: template first name
    template_first_name = get_secret("MONDAY_TEMPLATE_FIRST_NAME")
    if template_first_name:
        config.template_first_name = template_first_name

    return config


# =============================================================================
# GOOGLE SHEETS SETTINGS PERSISTENCE
# =============================================================================

def _get_settings_worksheet():
    """Get or create the 'Settings' worksheet in the advisor spreadsheet.

    Returns:
        gspread Worksheet object, or None if Google Sheets is not configured.
    """
    try:
        from src.utils.advisor_matcher import GSHEETS_AVAILABLE, get_gcp_credentials, get_secret
    except ImportError:
        return None

    if not GSHEETS_AVAILABLE:
        return None

    spreadsheet_id = get_secret('GOOGLE_SHEETS_SPREADSHEET_ID')
    if not spreadsheet_id:
        return None

    credentials = get_gcp_credentials()
    if not credentials:
        return None

    import gspread
    try:
        client = gspread.authorize(credentials)
        spreadsheet = client.open_by_key(spreadsheet_id)

        try:
            worksheet = spreadsheet.worksheet("Settings")
        except gspread.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title="Settings", rows=20, cols=2)
            worksheet.update('A1:B1', [['key', 'value']])
            worksheet.format('A1:B1', {'textFormat': {'bold': True}})

        return worksheet
    except Exception as e:
        logger.warning(f"Failed to access Settings worksheet: {e}")
        return None


def get_last_group_update_month() -> Optional[str]:
    """Read the last month when next-month groups were created.

    Returns:
        "YYYY-MM" string, or None if never executed.
    """
    worksheet = _get_settings_worksheet()
    if not worksheet:
        return None

    try:
        records = worksheet.get_all_records()
        for row in records:
            if str(row.get('key', '')).strip() == 'last_group_update_month':
                value = str(row.get('value', '')).strip()
                return value if value else None
        return None
    except Exception as e:
        logger.warning(f"Failed to read last_group_update_month: {e}")
        return None


def set_last_group_update_month(year_month: str) -> bool:
    """Write the current month as last update month in Google Sheets.

    Args:
        year_month: "YYYY-MM" format string

    Returns:
        True if successfully written, False otherwise.
    """
    worksheet = _get_settings_worksheet()
    if not worksheet:
        return False

    try:
        records = worksheet.get_all_records()
        for idx, row in enumerate(records):
            if str(row.get('key', '')).strip() == 'last_group_update_month':
                # Update existing row (idx+2 because header is row 1, 0-indexed records)
                worksheet.update_cell(idx + 2, 2, year_month)
                return True

        # Key not found, append new row
        worksheet.append_row(['last_group_update_month', year_month])
        return True
    except Exception as e:
        logger.warning(f"Failed to write last_group_update_month: {e}")
        return False


# =============================================================================
# NEXT-MONTH GROUP PROVISIONING
# =============================================================================

def get_next_month_group_name() -> str:
    """Get the group name for the next month in French.

    Returns:
        e.g., "Avril 2026" (if today is March 2026)
    """
    today = date.today()
    if today.month == 12:
        next_month, next_year = 1, today.year + 1
    else:
        next_month, next_year = today.month + 1, today.year
    return f"{MONTHS_FR[next_month]} {next_year}"


@dataclass
class NextMonthGroupsResult:
    """Result of the next-month group provisioning."""
    total_boards: int = 0
    groups_created: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)
    skipped_same_month: bool = False


def ensure_next_month_groups() -> NextMonthGroupsResult:
    """Ensure next month's group exists on all advisor boards.

    Checks Google Sheets to see if we already ran this month.
    If not, iterates over all advisor folders and creates the
    next month's group on each board.

    Returns:
        NextMonthGroupsResult with stats about what was done.
    """
    result = NextMonthGroupsResult()
    current_month = date.today().strftime("%Y-%m")

    # Check if already done this month
    last_update = get_last_group_update_month()
    if last_update == current_month:
        result.skipped_same_month = True
        logger.info(f"Next-month groups already provisioned for {current_month}, skipping.")
        return result

    # Load provisioning config
    config = load_provisioning_config()
    if not config:
        result.errors.append("Provisioning config not available")
        logger.warning("Cannot provision next-month groups: config not available")
        return result

    # Get Monday client
    from src.app.state import get_secret
    api_key = get_secret("MONDAY_API_KEY")
    if not api_key:
        result.errors.append("MONDAY_API_KEY not available")
        return result

    try:
        client = MondayClient(api_key=api_key)
    except MondayError as e:
        result.errors.append(f"Failed to create Monday client: {e}")
        return result

    next_month_name = get_next_month_group_name()
    logger.info(f"Provisioning next-month groups: '{next_month_name}'")

    # List all folders in workspace with parent info
    try:
        all_folders = client.list_all_folders_in_workspace_sync(config.workspace_id)
    except MondayError as e:
        result.errors.append(f"Failed to list folders: {e}")
        return result

    # Filter to advisor folders (children of conseillers_folder_id)
    advisor_folders = [
        f for f in all_folders
        if f.get("parent") and str(f["parent"]["id"]) == str(config.conseillers_folder_id)
    ]

    if not advisor_folders:
        logger.info("No advisor folders found under conseillers folder.")
        set_last_group_update_month(current_month)
        return result

    logger.info(f"Found {len(advisor_folders)} advisor folder(s)")

    # For each advisor folder, list boards and ensure next month group
    for folder in advisor_folders:
        folder_id = int(folder["id"])
        folder_name = folder["name"]

        try:
            boards = client.list_boards_in_folder_sync(folder_id)
        except MondayError as e:
            result.errors.append(f"Failed to list boards in '{folder_name}': {e}")
            continue

        for board in boards:
            board_id = int(board["id"])
            result.total_boards += 1

            try:
                groups = client.list_groups_sync(board_id)
                group_exists = any(g["title"] == next_month_name for g in groups)

                if group_exists:
                    result.skipped += 1
                else:
                    client.get_or_create_group_sync(board_id, next_month_name)
                    result.groups_created += 1
                    logger.info(f"Created group '{next_month_name}' on board '{board['name']}'")
            except MondayError as e:
                result.errors.append(f"Failed on board '{board['name']}': {e}")

            # Rate limit between boards
            time.sleep(0.3)

    # Mark this month as done
    set_last_group_update_month(current_month)
    logger.info(
        f"Next-month provisioning complete: {result.groups_created} created, "
        f"{result.skipped} skipped, {len(result.errors)} error(s)"
    )

    return result


def rename_boards_in_advisor_folders(
    pattern: str,
    replacement: str,
) -> dict:
    """Rename boards across all advisor folders by replacing a pattern in board names.

    Args:
        pattern: String to find in board names (e.g., " 2025")
        replacement: String to replace with (e.g., "")

    Returns:
        Dict with stats: {total_boards, renamed, skipped, errors}
    """
    config = load_provisioning_config()
    if not config:
        raise RuntimeError("Provisioning config not available")

    from src.app.state import get_secret
    api_key = get_secret("MONDAY_API_KEY")
    if not api_key:
        raise RuntimeError("MONDAY_API_KEY not available")

    client = MondayClient(api_key=api_key)
    stats = {"total_boards": 0, "renamed": 0, "skipped": 0, "errors": []}

    # List all folders with parent info
    all_folders = client.list_all_folders_in_workspace_sync(config.workspace_id)
    advisor_folders = [
        f for f in all_folders
        if f.get("parent") and str(f["parent"]["id"]) == str(config.conseillers_folder_id)
    ]

    for folder in advisor_folders:
        folder_id = int(folder["id"])
        try:
            boards = client.list_boards_in_folder_sync(folder_id)
        except MondayError as e:
            stats["errors"].append(f"Failed to list boards in '{folder['name']}': {e}")
            continue

        for board in boards:
            stats["total_boards"] += 1
            board_name = board["name"]

            if pattern in board_name:
                new_name = board_name.replace(pattern, replacement)
                try:
                    client.update_board_name_sync(int(board["id"]), new_name)
                    stats["renamed"] += 1
                    logger.info(f"Renamed '{board_name}' → '{new_name}'")
                except MondayError as e:
                    stats["errors"].append(f"Failed to rename '{board_name}': {e}")
                time.sleep(0.3)
            else:
                stats["skipped"] += 1

    return stats


# =============================================================================
# PROVISIONER
# =============================================================================

class AdvisorBoardProvisioner:
    """Orchestrates Monday.com board provisioning for new advisors."""

    def __init__(self, client: MondayClient, config: ProvisioningConfig):
        self.client = client
        self.config = config

    def provision(
        self,
        advisor,
        progress_callback: Optional[Callable[[str, str], None]] = None,
    ) -> ProvisioningResult:
        """Provision Monday.com boards for a new advisor.

        6-step process:
        1. Create advisor folder
        2. Discover template boards
        3. Duplicate and rename each board
        4. Clean up groups (delete old, create current month)
        5. Invite advisor via email (if email provided)
        6. Set board permissions to contributor (edit content only)

        Args:
            advisor: Advisor dataclass with first_name, last_name, email
            progress_callback: Optional callback(step_name, message)

        Returns:
            ProvisioningResult with details of what was created
        """
        result = ProvisioningResult()
        advisor_name = f"{advisor.first_name} {advisor.last_name}"

        def _update(step_name: str, message: str):
            if progress_callback:
                progress_callback(step_name, message)

        # Step 1: Create advisor folder
        step1 = ProvisioningStep(name="Créer le dossier")
        result.steps.append(step1)
        step1.status = "running"
        _update("Créer le dossier", f"Création du dossier pour {advisor_name}...")

        try:
            folder = self.client.create_folder_sync(
                name=advisor_name,
                workspace_id=self.config.workspace_id,
                parent_folder_id=self.config.conseillers_folder_id,
            )
            result.advisor_folder_id = folder["id"]
            step1.status = "success"
            step1.message = f"Dossier créé (ID: {folder['id']})"
            logger.info(f"Created folder '{advisor_name}' with ID {folder['id']}")
        except MondayError as e:
            step1.status = "error"
            step1.message = str(e)
            result.errors.append(f"Folder creation failed: {e}")
            logger.error(f"Failed to create folder for {advisor_name}: {e}")
            return result

        # Step 2: Discover template boards
        step2 = ProvisioningStep(name="Découvrir les templates")
        result.steps.append(step2)
        step2.status = "running"
        _update("Découvrir les templates", "Recherche des boards template...")

        template_boards = []
        try:
            if self.config.template_board_ids:
                # Use explicit board IDs
                template_boards = [
                    {"id": str(bid), "name": f"Board {bid}"}
                    for bid in self.config.template_board_ids
                ]
            else:
                # Auto-discover from folder
                template_boards = self.client.list_boards_in_folder_sync(
                    self.config.template_folder_id
                )

            if not template_boards:
                step2.status = "error"
                step2.message = "Aucun board template trouvé"
                result.errors.append("No template boards found")
                return result

            step2.status = "success"
            step2.message = f"{len(template_boards)} board(s) trouvé(s)"
            logger.info(f"Found {len(template_boards)} template boards")
        except MondayError as e:
            step2.status = "error"
            step2.message = str(e)
            result.errors.append(f"Template discovery failed: {e}")
            logger.error(f"Failed to discover template boards: {e}")
            return result

        # Step 3: Duplicate and rename each board
        step3 = ProvisioningStep(name="Dupliquer les boards")
        result.steps.append(step3)
        step3.status = "running"

        new_folder_id = int(result.advisor_folder_id)
        duplicated_boards = []

        for i, template in enumerate(template_boards):
            template_id = int(template["id"])
            template_name = template["name"]
            new_name = rename_board_for_advisor(
                template_name,
                advisor.first_name,
                self.config.template_first_name,
            )

            _update(
                "Dupliquer les boards",
                f"Duplication {i + 1}/{len(template_boards)}: {new_name}...",
            )

            try:
                new_board = self.client.duplicate_board_sync(
                    board_id=template_id,
                    board_name=new_name,
                    folder_id=new_folder_id,
                    workspace_id=self.config.workspace_id,
                )
                new_board_id = new_board["id"]

                # Rename if the duplication didn't use the name correctly
                actual_name = new_board.get("name", "")
                if actual_name != new_name:
                    self.client.update_board_name_sync(int(new_board_id), new_name)

                duplicated_boards.append({
                    "id": new_board_id,
                    "name": new_name,
                })
                result.created_board_ids.append(new_board_id)
                logger.info(f"Duplicated board '{template_name}' → '{new_name}' (ID: {new_board_id})")

                # Rate limit delay between duplications
                if i < len(template_boards) - 1:
                    time.sleep(DUPLICATION_DELAY)

            except MondayError as e:
                result.errors.append(f"Board duplication failed for '{template_name}': {e}")
                logger.error(f"Failed to duplicate board '{template_name}': {e}")

        if duplicated_boards:
            step3.status = "success"
            step3.message = f"{len(duplicated_boards)} board(s) dupliqué(s)"
        else:
            step3.status = "error"
            step3.message = "Aucun board dupliqué"
            return result

        # Step 4: Clean up groups
        step4 = ProvisioningStep(name="Nettoyer les groupes")
        result.steps.append(step4)
        step4.status = "running"
        current_month = get_current_month_group_name()
        _update("Nettoyer les groupes", f"Configuration des groupes ({current_month})...")

        groups_cleaned = 0
        for board_info in duplicated_boards:
            board_id = int(board_info["id"])
            try:
                # List and delete all existing groups
                groups = self.client.list_groups_sync(board_id)
                for group in groups:
                    try:
                        self.client.delete_group_sync(board_id, group["id"])
                    except MondayError as e:
                        logger.warning(f"Failed to delete group '{group['title']}': {e}")

                # Create current month group
                self.client.get_or_create_group_sync(board_id, current_month)
                groups_cleaned += 1
                time.sleep(DUPLICATION_DELAY)
            except MondayError as e:
                result.errors.append(f"Group cleanup failed for board {board_id}: {e}")
                logger.error(f"Failed to clean groups for board {board_id}: {e}")

        step4.status = "success" if groups_cleaned > 0 else "error"
        step4.message = f"{groups_cleaned}/{len(duplicated_boards)} board(s) nettoyé(s)"

        # Step 5: Invite advisor (if email provided)
        step5 = ProvisioningStep(name="Inviter le conseiller")
        result.steps.append(step5)

        if advisor.email:
            step5.status = "running"
            _update("Inviter le conseiller", f"Invitation de {advisor.email}...")

            try:
                invite_result = self.client.invite_users_sync([advisor.email])
                invited_users = invite_result.get("invited_users", [])
                invite_errors = invite_result.get("errors", [])

                if invited_users:
                    user_id = int(invited_users[0]["id"])
                    result.invited_user_id = user_id

                    # Add user to each duplicated board
                    for board_info in duplicated_boards:
                        try:
                            self.client.add_users_to_board_sync(
                                board_id=int(board_info["id"]),
                                user_ids=[user_id],
                                kind="subscriber",
                            )
                        except MondayError as e:
                            logger.warning(
                                f"Failed to add user to board {board_info['id']}: {e}"
                            )

                    step5.status = "success"
                    step5.message = f"Invitation envoyée à {advisor.email}"
                    logger.info(f"Invited {advisor.email} (user ID: {user_id})")
                elif invite_errors:
                    error_msgs = [e.get("message", "") for e in invite_errors]
                    step5.status = "error"
                    step5.message = "; ".join(error_msgs)
                    result.errors.append(f"Invitation failed: {'; '.join(error_msgs)}")
                else:
                    step5.status = "success"
                    step5.message = "Invitation envoyée (utilisateur peut-être déjà existant)"

            except MondayError as e:
                step5.status = "error"
                step5.message = str(e)
                result.errors.append(f"Invitation failed: {e}")
                logger.error(f"Failed to invite {advisor.email}: {e}")
        else:
            step5.status = "success"
            step5.message = "Pas d'email fourni, invitation ignorée"

        # Step 6: Set board permissions to contributor
        step6 = ProvisioningStep(name="Configurer les permissions")
        result.steps.append(step6)
        step6.status = "running"
        _update("Configurer les permissions", "Restriction des boards en mode contributeur...")

        permissions_set = 0
        for board_info in duplicated_boards:
            board_id = int(board_info["id"])
            try:
                self.client.set_board_permission_sync(
                    board_id=board_id,
                    basic_role_name="contributor",
                )
                permissions_set += 1
                logger.info(f"Set board {board_info['name']} to contributor mode")
            except MondayError as e:
                logger.warning(f"Failed to set permissions on board {board_info['id']}: {e}")
                result.errors.append(f"Permission setting failed for board {board_info['name']}: {e}")
            time.sleep(0.3)

        if permissions_set > 0:
            step6.status = "success"
            step6.message = f"{permissions_set}/{len(duplicated_boards)} board(s) en mode contributeur"
        else:
            step6.status = "error"
            step6.message = "Aucun board configuré (plan Enterprise requis ?)"

        # Final status
        critical_steps = [s for s in result.steps if s.name in ["Créer le dossier", "Dupliquer les boards"]]
        result.success = all(s.status == "success" for s in critical_steps)

        return result
