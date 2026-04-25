/**
 * Remove all currently selected objects from the hierarchy.
 * Run from Automate → Show script editor, or bind as a custom command.
 */
def hier = getCurrentHierarchy()
def sel = new ArrayList<>(getSelectedObjects())
if (sel.isEmpty()) {
    print("No objects selected.")
    return
}
hier.removeObjects(sel, true)
hier.getSelectionModel().clearSelection()
fireHierarchyUpdate()
print("Removed ${sel.size()} object(s).")
