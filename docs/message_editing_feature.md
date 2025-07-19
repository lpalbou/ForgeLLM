# Message Editing Feature

## Overview

The message editing feature allows users to edit their previous messages in the testing tab's chat interface. When a message is edited, all subsequent messages in the conversation are removed, and a new response is generated from the edited message forward.

## Features

### ✨ Key Capabilities

1. **Edit Button on Hover**: Edit buttons appear when hovering over user messages
2. **In-place Editing**: Edit messages directly in the chat interface
3. **History Truncation**: Automatically removes all messages after the edited message
4. **Smart Warnings**: Shows how many messages will be removed before editing
5. **Token Count Updates**: Maintains accurate token counting after edits
6. **Keyboard Shortcuts**: Ctrl+Enter to save, Escape to cancel
7. **Confirmation Dialogs**: Asks for confirmation when edits will remove messages
8. **Save/Load Compatibility**: Edited conversations are preserved in saved history

### 🎯 User Experience

- **Visual Feedback**: Edit buttons fade in/out smoothly on hover
- **Clear Interface**: Inline editing with intuitive controls
- **Safety First**: Warns users about destructive operations
- **Keyboard Friendly**: Full keyboard navigation support
- **Non-disruptive**: Doesn't interfere with ongoing generation

## How It Works

### 1. Edit Button Visibility
```javascript
// Edit buttons appear on hover with smooth opacity transition
userBubble.addEventListener('mouseenter', () => {
    editContainer.style.opacity = '1';
});

userBubble.addEventListener('mouseleave', () => {
    editContainer.style.opacity = '0';
});
```

### 2. Edit Interface
When the edit button is clicked:
- Original message content is replaced with an edit interface
- Textarea is pre-filled with the current message text
- Warning shows how many messages will be removed (if any)
- Cancel and Save buttons provide clear actions

### 3. History Truncation
```javascript
// Calculate messages to remove
const allMessages = Array.from(chatHistory.children);
const messagesToRemove = allMessages.slice(messageIndex + 1);

// Remove messages after the edited one
messagesToRemove.forEach(message => message.remove());
```

### 4. Token Count Management
- Calculates tokens to subtract from removed messages
- Updates token difference from original vs edited message
- Maintains accurate conversation token totals

## Technical Implementation

### Core Functions

#### `addEditButtonToUserMessage(userBubble, originalPrompt)`
- Adds edit button to user messages
- Sets up hover events and click handlers
- Creates smooth opacity transitions

#### `startEditingMessage(userBubble, originalPrompt)`
- Creates edit interface with textarea and controls
- Calculates and displays truncation warnings
- Sets up keyboard shortcuts and event handlers
- Disables chat input during editing

#### `saveEditedMessage(userBubble, newPrompt, messageIndex)`
- Truncates conversation history from edit point
- Updates token counts accurately
- Regenerates response from edited message
- Re-enables interface elements

### Integration Points

#### Generation Flow
- Edit buttons are disabled during text generation
- Chat input is disabled while editing
- Buttons are re-enabled when generation completes

#### History Management
- Edited messages are stored with `data-original-prompt` attribute
- Save/load functionality preserves edited content
- Token counts remain accurate across sessions

#### User Interface
- Edit interface uses glassmorphism styling
- Buttons have hover effects and proper contrast
- Warning messages are color-coded (yellow for warnings)

## CSS Styling

### Edit Controls
```css
.message-edit-controls {
    margin-top: 8px !important;
    transition: opacity 0.2s ease;
}

.message-edit-controls .btn {
    background: rgba(255, 255, 255, 0.2) !important;
    backdrop-filter: blur(4px);
}
```

### Edit Interface
```css
.message-edit-interface {
    background: rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
}
```

## Usage Examples

### Basic Edit
1. Hover over a user message
2. Click the edit button (pencil icon)
3. Modify the text in the textarea
4. Press Ctrl+Enter or click "Save & Continue"
5. New response is generated from the edited message

### Edit with History Truncation
1. Edit a message that has responses after it
2. Warning appears: "This will remove X messages after this point"
3. Confirmation dialog asks if you want to continue
4. After confirmation, subsequent messages are removed
5. New conversation continues from the edited message

### Canceling an Edit
1. Start editing a message
2. Press Escape or click "Cancel"
3. Original message is restored unchanged

## Best Practices

### For Users
- ✅ Edit messages to refine prompts and get better responses
- ✅ Use editing to fix typos or clarify questions
- ✅ Be aware that editing removes subsequent conversation
- ❌ Don't edit if you want to keep the existing conversation flow

### For Developers
- ✅ Always update token counts when modifying history
- ✅ Disable edit functionality during generation
- ✅ Provide clear warnings for destructive operations
- ✅ Maintain keyboard accessibility

## Testing

### Manual Testing Steps
1. Load a model and have a conversation with multiple exchanges
2. Hover over user messages to verify edit buttons appear
3. Test editing the last message (no truncation)
4. Test editing an earlier message (with truncation warning)
5. Test canceling edits with both buttons and keyboard
6. Verify token counts update correctly
7. Test save/load with edited conversations

### Automated Tests
See `tests/test_message_editing.py` for Selenium-based automated tests that verify:
- Edit button visibility
- Edit interface functionality
- Cancellation behavior
- Keyboard shortcuts
- History truncation warnings

## Future Enhancements

### Potential Improvements
- **Edit History**: Track edit history for undo/redo
- **Batch Editing**: Edit multiple messages at once
- **Edit Assistant Messages**: Allow editing of AI responses
- **Diff View**: Show changes between original and edited messages
- **Auto-save Drafts**: Save edit drafts automatically

### Technical Considerations
- **Performance**: Optimize for conversations with many messages
- **Accessibility**: Improve screen reader support
- **Mobile**: Enhance touch interaction for mobile devices
- **Collaboration**: Support for multi-user editing scenarios

## Conclusion

The message editing feature provides a powerful way for users to refine their conversations with AI models. By allowing edits while maintaining conversation integrity through smart truncation, users can iteratively improve their prompts and explore different conversation paths efficiently.

The implementation prioritizes user safety with clear warnings, maintains technical accuracy with proper token counting, and provides an intuitive interface that integrates seamlessly with the existing chat system. 