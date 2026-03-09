"""
Tests for advisor board provisioning.

Usage:
    python -m src.tests.test_advisor_provisioning            # Run all unit tests
    python -m src.tests.test_advisor_provisioning unit        # Run unit tests only
    python -m src.tests.test_advisor_provisioning integration # Run integration tests (requires API)
    python -m src.tests.test_advisor_provisioning discover    # Run discovery script
"""

import os
import sys
import unittest
from datetime import date
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.clients.monday import MondayClient
from src.utils.advisor_provisioning import (
    AdvisorBoardProvisioner,
    ProvisioningConfig,
    get_current_month_group_name,
    load_provisioning_config,
    rename_board_for_advisor,
)
from src.utils.aggregator import MONTHS_FR

# =============================================================================
# UNIT TESTS
# =============================================================================

class TestRenameBoardForAdvisor(unittest.TestCase):
    """Test board renaming logic."""

    def test_standard_rename(self):
        result = rename_board_for_advisor("Thomas - Ventes", "Jean")
        self.assertEqual(result, "Jean - Ventes")

    def test_rename_with_different_template(self):
        result = rename_board_for_advisor("Marie - Paiement Historique", "Pierre", "Marie")
        self.assertEqual(result, "Pierre - Paiement Historique")

    def test_rename_preserves_suffix(self):
        result = rename_board_for_advisor("Thomas - Ventes et Production", "Alice")
        self.assertEqual(result, "Alice - Ventes et Production")

    def test_rename_no_separator(self):
        """Board without ' - ' should get advisor prefix."""
        result = rename_board_for_advisor("Some Board", "Jean")
        self.assertEqual(result, "Jean - Some Board")

    def test_rename_multiple_separators(self):
        """Only first ' - ' should be used for splitting."""
        result = rename_board_for_advisor("Thomas - AE - Tracker", "Jean")
        self.assertEqual(result, "Jean - AE - Tracker")

    def test_rename_empty_advisor(self):
        result = rename_board_for_advisor("Thomas - Ventes", "")
        self.assertEqual(result, " - Ventes")


class TestGetCurrentMonthGroupName(unittest.TestCase):
    """Test current month group name generation."""

    def test_returns_french_month(self):
        result = get_current_month_group_name()
        today = date.today()
        expected = f"{MONTHS_FR[today.month]} {today.year}"
        self.assertEqual(result, expected)

    @patch("src.utils.advisor_provisioning.date")
    def test_february_2026(self, mock_date):
        mock_date.today.return_value = date(2026, 2, 15)
        result = get_current_month_group_name()
        self.assertEqual(result, "Février 2026")

    @patch("src.utils.advisor_provisioning.date")
    def test_december(self, mock_date):
        mock_date.today.return_value = date(2025, 12, 1)
        result = get_current_month_group_name()
        self.assertEqual(result, "Décembre 2025")


class TestProvisioningConfigLoading(unittest.TestCase):
    """Test config loading from environment."""

    @patch("src.app.state.get_secret")
    def test_valid_config(self, mock_get_secret):
        """All required values present."""
        mock_get_secret.side_effect = lambda key, default=None: {
            "MONDAY_WORKSPACE_ID": "12345",
            "MONDAY_CONSEILLERS_FOLDER_ID": "67890",
            "MONDAY_TEMPLATE_FOLDER_ID": "11111",
            "MONDAY_TEMPLATE_BOARD_IDS": None,
            "MONDAY_TEMPLATE_FIRST_NAME": None,
        }.get(key, default)

        config = load_provisioning_config()

        self.assertIsNotNone(config)
        self.assertEqual(config.workspace_id, 12345)
        self.assertEqual(config.conseillers_folder_id, 67890)
        self.assertEqual(config.template_folder_id, 11111)
        self.assertIsNone(config.template_board_ids)
        self.assertEqual(config.template_first_name, "Thomas")

    @patch("src.app.state.get_secret")
    def test_missing_workspace_id(self, mock_get_secret):
        """Missing required value returns None."""
        mock_get_secret.side_effect = lambda key, default=None: {
            "MONDAY_WORKSPACE_ID": None,
            "MONDAY_CONSEILLERS_FOLDER_ID": "67890",
            "MONDAY_TEMPLATE_FOLDER_ID": "11111",
        }.get(key, default)

        config = load_provisioning_config()
        self.assertIsNone(config)

    @patch("src.app.state.get_secret")
    def test_config_with_board_ids(self, mock_get_secret):
        """Explicit board IDs are parsed correctly."""
        mock_get_secret.side_effect = lambda key, default=None: {
            "MONDAY_WORKSPACE_ID": "12345",
            "MONDAY_CONSEILLERS_FOLDER_ID": "67890",
            "MONDAY_TEMPLATE_FOLDER_ID": "11111",
            "MONDAY_TEMPLATE_BOARD_IDS": "100,200,300",
            "MONDAY_TEMPLATE_FIRST_NAME": "Marie",
        }.get(key, default)

        config = load_provisioning_config()

        self.assertIsNotNone(config)
        self.assertEqual(config.template_board_ids, [100, 200, 300])
        self.assertEqual(config.template_first_name, "Marie")

    @patch("src.app.state.get_secret")
    def test_invalid_numeric_values(self, mock_get_secret):
        """Non-numeric values return None."""
        mock_get_secret.side_effect = lambda key, default=None: {
            "MONDAY_WORKSPACE_ID": "not_a_number",
            "MONDAY_CONSEILLERS_FOLDER_ID": "67890",
            "MONDAY_TEMPLATE_FOLDER_ID": "11111",
        }.get(key, default)

        from src.utils.advisor_provisioning import load_provisioning_config
        config = load_provisioning_config()
        self.assertIsNone(config)


class TestProvisionerWithMocks(unittest.TestCase):
    """Test provisioner logic with mocked Monday.com client."""

    def _make_advisor(self, first_name="Jean", last_name="Dupont", email="jean@test.com"):
        """Create a mock advisor."""
        advisor = MagicMock()
        advisor.first_name = first_name
        advisor.last_name = last_name
        advisor.email = email
        return advisor

    def _make_config(self):
        return ProvisioningConfig(
            workspace_id=12345,
            conseillers_folder_id=67890,
            template_folder_id=11111,
            template_first_name="Thomas",
        )

    def test_full_provisioning_success(self):
        """Test successful end-to-end provisioning with mocks."""
        client = MagicMock()
        client.create_folder_sync.return_value = {"id": "999"}
        client.list_boards_in_folder_sync.return_value = [
            {"id": "100", "name": "Thomas - Ventes"},
            {"id": "101", "name": "Thomas - Paiement Historique"},
        ]
        client.duplicate_board_sync.side_effect = [
            {"id": "200", "name": "Jean - Ventes"},
            {"id": "201", "name": "Jean - Paiement Historique"},
        ]
        client.list_groups_sync.return_value = [
            {"id": "g1", "title": "Janvier 2026"},
        ]
        client.delete_group_sync.return_value = {"id": "g1", "deleted": True}
        client.get_or_create_group_sync.return_value = MagicMock(success=True, id="g_new")
        client.invite_users_sync.return_value = {
            "invited_users": [{"id": 42, "email": "jean@test.com"}],
            "errors": [],
        }
        client.add_users_to_board_sync.return_value = {"id": "200"}

        config = self._make_config()
        provisioner = AdvisorBoardProvisioner(client, config)
        advisor = self._make_advisor()

        result = provisioner.provision(advisor)

        self.assertTrue(result.success)
        self.assertEqual(result.advisor_folder_id, "999")
        self.assertEqual(len(result.created_board_ids), 2)
        self.assertEqual(result.invited_user_id, 42)
        self.assertEqual(len(result.errors), 0)

        # Verify folder was created under conseillers
        client.create_folder_sync.assert_called_once_with(
            name="Jean Dupont",
            workspace_id=12345,
            parent_folder_id=67890,
        )

    def test_provisioning_no_email(self):
        """Provisioning works without email (skip invitation)."""
        client = MagicMock()
        client.create_folder_sync.return_value = {"id": "999"}
        client.list_boards_in_folder_sync.return_value = [
            {"id": "100", "name": "Thomas - Ventes"},
        ]
        client.duplicate_board_sync.return_value = {"id": "200", "name": "Jean - Ventes"}
        client.list_groups_sync.return_value = []
        client.get_or_create_group_sync.return_value = MagicMock(success=True)

        config = self._make_config()
        provisioner = AdvisorBoardProvisioner(client, config)
        advisor = self._make_advisor(email=None)

        result = provisioner.provision(advisor)

        self.assertTrue(result.success)
        self.assertIsNone(result.invited_user_id)
        client.invite_users_sync.assert_not_called()

    def test_provisioning_folder_creation_fails(self):
        """If folder creation fails, provisioning stops early."""
        from src.clients.monday import MondayError

        client = MagicMock()
        client.create_folder_sync.side_effect = MondayError("API error")

        config = self._make_config()
        provisioner = AdvisorBoardProvisioner(client, config)
        advisor = self._make_advisor()

        result = provisioner.provision(advisor)

        self.assertFalse(result.success)
        self.assertTrue(len(result.errors) > 0)
        self.assertEqual(len(result.created_board_ids), 0)

    def test_provisioning_with_explicit_board_ids(self):
        """Config with explicit template_board_ids skips folder discovery."""
        client = MagicMock()
        client.create_folder_sync.return_value = {"id": "999"}
        client.duplicate_board_sync.return_value = {"id": "200", "name": "Jean - Board"}
        client.list_groups_sync.return_value = []
        client.get_or_create_group_sync.return_value = MagicMock(success=True)

        config = self._make_config()
        config.template_board_ids = [100, 101]
        provisioner = AdvisorBoardProvisioner(client, config)
        advisor = self._make_advisor(email=None)

        result = provisioner.provision(advisor)

        self.assertTrue(result.success)
        self.assertEqual(len(result.created_board_ids), 2)
        # Should NOT have called list_boards_in_folder
        client.list_boards_in_folder_sync.assert_not_called()


# =============================================================================
# INTEGRATION TESTS (requires MONDAY_API_KEY)
# =============================================================================

class TestIntegrationMondayAPI(unittest.TestCase):
    """Integration tests that call the real Monday.com API.

    Only run with: python -m src.tests.test_advisor_provisioning integration
    """

    @classmethod
    def setUpClass(cls):
        api_key = os.environ.get("MONDAY_API_KEY")
        if not api_key:
            raise unittest.SkipTest("MONDAY_API_KEY not set")
        cls.client = MondayClient(api_key=api_key)

    def test_list_folders(self):
        """List folders in all workspaces."""
        result = self.client._execute_query_sync("{ workspaces { id name } }")
        workspaces = result["data"]["workspaces"]
        self.assertTrue(len(workspaces) > 0, "Should have at least one workspace")

        for ws in workspaces:
            folders = self.client.list_folders_sync(int(ws["id"]))
            print(f"Workspace '{ws['name']}': {len(folders)} folder(s)")
            for f in folders:
                print(f"  - {f['name']} (ID: {f['id']})")

    def test_list_boards_in_folder(self):
        """List boards in the template folder (if configured)."""
        template_folder_id = os.environ.get("MONDAY_TEMPLATE_FOLDER_ID")
        if not template_folder_id:
            self.skipTest("MONDAY_TEMPLATE_FOLDER_ID not set")

        boards = self.client.list_boards_in_folder_sync(int(template_folder_id))
        print(f"Template folder has {len(boards)} board(s):")
        for b in boards:
            print(f"  - {b['name']} (ID: {b['id']})")
        self.assertTrue(len(boards) > 0, "Template folder should have boards")


# =============================================================================
# MAIN
# =============================================================================

def run_unit_tests():
    """Run unit tests only."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestRenameBoardForAdvisor))
    suite.addTests(loader.loadTestsFromTestCase(TestGetCurrentMonthGroupName))
    suite.addTests(loader.loadTestsFromTestCase(TestProvisioningConfigLoading))
    suite.addTests(loader.loadTestsFromTestCase(TestProvisionerWithMocks))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


def run_integration_tests():
    """Run integration tests (requires API key)."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestIntegrationMondayAPI)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


def run_discover():
    """Run discovery script."""
    from src.tests.discover_monday_ids import discover
    discover()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "unit"

    if mode == "unit":
        success = run_unit_tests()
    elif mode == "integration":
        success = run_integration_tests()
    elif mode == "discover":
        run_discover()
        success = True
    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python -m src.tests.test_advisor_provisioning [unit|integration|discover]")
        success = False

    sys.exit(0 if success else 1)
