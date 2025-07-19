#!/usr/bin/env python3
"""
Test script for message editing functionality in the testing tab.

This test demonstrates the expected behavior of the message edit feature:
1. User messages should have an edit button that appears on hover
2. Clicking edit should show an edit interface
3. Editing a message should truncate history from that point
4. The edited message should generate a new response
5. Token counts should be updated correctly
"""

import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys


class MessageEditingTest:
    def __init__(self):
        self.driver = None
        self.wait = None
    
    def setup(self):
        """Setup the browser and navigate to the application."""
        # You can use Chrome, Firefox, etc.
        self.driver = webdriver.Chrome()
        self.wait = WebDriverWait(self.driver, 10)
        
        # Navigate to the application
        self.driver.get("http://localhost:5000")
        
        # Switch to testing tab
        testing_tab = self.wait.until(
            EC.element_to_be_clickable((By.ID, "testing-tab"))
        )
        testing_tab.click()
    
    def test_edit_button_visibility(self):
        """Test that edit buttons appear on hover for user messages."""
        print("Testing edit button visibility...")
        
        # Wait for a user message to be present
        user_message = self.wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".chat-user"))
        )
        
        # Hover over the message
        ActionChains(self.driver).move_to_element(user_message).perform()
        
        # Check that edit button becomes visible
        edit_button = self.wait.until(
            EC.visibility_of_element_located(
                (By.CSS_SELECTOR, ".chat-user .message-edit-controls button")
            )
        )
        
        assert edit_button.is_displayed(), "Edit button should be visible on hover"
        print("✓ Edit button visibility test passed")
    
    def test_edit_interface(self):
        """Test that clicking edit shows the edit interface."""
        print("Testing edit interface...")
        
        # Find a user message and hover over it
        user_message = self.driver.find_element(By.CSS_SELECTOR, ".chat-user")
        ActionChains(self.driver).move_to_element(user_message).perform()
        
        # Click the edit button
        edit_button = self.driver.find_element(
            By.CSS_SELECTOR, ".chat-user .message-edit-controls button"
        )
        edit_button.click()
        
        # Check that edit interface appears
        edit_interface = self.wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".message-edit-interface"))
        )
        
        # Check that textarea contains the original message
        textarea = edit_interface.find_element(By.CSS_SELECTOR, ".edit-textarea")
        assert textarea.get_attribute("value"), "Textarea should contain original message"
        
        # Check that cancel and save buttons are present
        cancel_btn = edit_interface.find_element(By.CSS_SELECTOR, ".cancel-edit-btn")
        save_btn = edit_interface.find_element(By.CSS_SELECTOR, ".save-edit-btn")
        
        assert cancel_btn.is_displayed(), "Cancel button should be visible"
        assert save_btn.is_displayed(), "Save button should be visible"
        
        print("✓ Edit interface test passed")
    
    def test_edit_cancellation(self):
        """Test that canceling edit restores the original message."""
        print("Testing edit cancellation...")
        
        # Get original message text
        user_message = self.driver.find_element(By.CSS_SELECTOR, ".chat-user")
        original_text = user_message.text
        
        # Start editing
        ActionChains(self.driver).move_to_element(user_message).perform()
        edit_button = user_message.find_element(
            By.CSS_SELECTOR, ".message-edit-controls button"
        )
        edit_button.click()
        
        # Click cancel
        cancel_btn = self.driver.find_element(By.CSS_SELECTOR, ".cancel-edit-btn")
        cancel_btn.click()
        
        # Check that original message is restored
        time.sleep(0.5)  # Wait for transition
        restored_text = user_message.text
        assert original_text in restored_text, "Original message should be restored"
        
        print("✓ Edit cancellation test passed")
    
    def test_keyboard_shortcuts(self):
        """Test keyboard shortcuts for editing."""
        print("Testing keyboard shortcuts...")
        
        # Start editing
        user_message = self.driver.find_element(By.CSS_SELECTOR, ".chat-user")
        ActionChains(self.driver).move_to_element(user_message).perform()
        edit_button = user_message.find_element(
            By.CSS_SELECTOR, ".message-edit-controls button"
        )
        edit_button.click()
        
        # Test Escape key cancellation
        textarea = self.driver.find_element(By.CSS_SELECTOR, ".edit-textarea")
        textarea.send_keys(Keys.ESCAPE)
        
        # Should restore original message
        time.sleep(0.5)
        assert not self.driver.find_elements(By.CSS_SELECTOR, ".message-edit-interface"), \
            "Edit interface should be closed after Escape"
        
        print("✓ Keyboard shortcuts test passed")
    
    def test_history_truncation_warning(self):
        """Test that users are warned about history truncation."""
        print("Testing history truncation warning...")
        
        # This test assumes there are multiple messages in the conversation
        user_messages = self.driver.find_elements(By.CSS_SELECTOR, ".chat-user")
        if len(user_messages) < 2:
            print("⚠ Skipping history truncation test - need multiple messages")
            return
        
        # Edit an earlier message (not the last one)
        first_message = user_messages[0]
        ActionChains(self.driver).move_to_element(first_message).perform()
        edit_button = first_message.find_element(
            By.CSS_SELECTOR, ".message-edit-controls button"
        )
        edit_button.click()
        
        # Check for warning text
        edit_interface = self.driver.find_element(By.CSS_SELECTOR, ".message-edit-interface")
        warning_text = edit_interface.text
        
        assert "remove" in warning_text.lower(), "Should show warning about removing messages"
        assert "message" in warning_text.lower(), "Should mention messages being removed"
        
        print("✓ History truncation warning test passed")
    
    def run_all_tests(self):
        """Run all tests."""
        print("Starting message editing functionality tests...")
        
        try:
            self.setup()
            
            # Note: These tests assume a model is loaded and there are existing messages
            # In a real test environment, you would set up the test data first
            
            self.test_edit_button_visibility()
            self.test_edit_interface()
            self.test_edit_cancellation()
            self.test_keyboard_shortcuts()
            self.test_history_truncation_warning()
            
            print("\n✅ All message editing tests passed!")
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            
        finally:
            if self.driver:
                self.driver.quit()


def manual_test_instructions():
    """
    Manual testing instructions for the message editing feature.
    """
    print("""
    📋 MANUAL TESTING INSTRUCTIONS FOR MESSAGE EDITING
    
    1. Setup:
       - Start the application: python -m forgellm.web.run
       - Navigate to http://localhost:5000
       - Go to the Testing tab
       - Load a model and have a conversation with a few messages
    
    2. Test Edit Button Visibility:
       - Hover over any user message (blue bubbles)
       - An edit button (pencil icon) should appear in the bottom right
       - Move mouse away, button should fade out
    
    3. Test Edit Interface:
       - Click the edit button on a user message
       - Should see:
         * Edit interface with textarea containing original message
         * Warning about removing subsequent messages (if any)
         * Cancel and Save buttons
         * Keyboard shortcut hint (Ctrl+Enter to save, Esc to cancel)
    
    4. Test Editing:
       - Modify the text in the textarea
       - Press Ctrl+Enter or click "Save & Continue"
       - If there are messages after this one, should get confirmation dialog
       - After confirmation, should remove subsequent messages and generate new response
    
    5. Test Cancellation:
       - Start editing a message
       - Press Escape or click Cancel
       - Should restore original message without changes
    
    6. Test Token Count Updates:
       - Note the token count before editing
       - Edit a message to be shorter/longer
       - Token count should update appropriately
    
    7. Test Save/Load History:
       - Edit some messages in a conversation
       - Save the conversation history
       - Load it back - edited messages should be preserved
    
    8. Edge Cases to Test:
       - Edit the last message (no subsequent messages to remove)
       - Edit with empty content (should show warning)
       - Edit while model is generating (buttons should be disabled)
       - Edit with very long text
    
    ✅ Expected Behaviors:
    - Edit buttons only appear on user messages, not assistant messages
    - Editing preserves conversation flow naturally
    - Token counts remain accurate
    - History truncation is clearly communicated
    - Keyboard shortcuts work reliably
    - No interference with ongoing generation
    """)


if __name__ == "__main__":
    print("Message Editing Feature Test")
    print("=" * 50)
    
    manual_test_instructions()
    
    # Uncomment to run automated tests (requires selenium setup)
    # test = MessageEditingTest()
    # test.run_all_tests() 