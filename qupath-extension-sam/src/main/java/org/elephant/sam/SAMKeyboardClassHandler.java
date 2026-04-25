package org.elephant.sam;

import java.util.ArrayList;
import java.util.List;

import javafx.event.EventHandler;
import javafx.scene.Node;
import javafx.scene.Scene;
import javafx.scene.control.TextInputControl;
import javafx.scene.input.KeyCode;
import javafx.scene.input.KeyEvent;
import javafx.stage.WindowEvent;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.objects.hierarchy.PathObjectHierarchy;

/**
 * When the main window has focus, digit keys 1–9 assign fixed {@link PathClass} names
 * (see {@link SAMAnnotClassShortcuts}) to all selected objects. Ignores typing in text fields.
 */
final class SAMKeyboardClassHandler {

	private final QuPathGUI qupath;

	private final EventHandler<KeyEvent> keyFilter = this::handleKeyPressed;

	private final EventHandler<WindowEvent> showingHandler = e -> attachToSceneIfNeeded();

	SAMKeyboardClassHandler(QuPathGUI qupath) {
		this.qupath = qupath;
	}

	void install() {
		var stage = qupath.getStage();
		stage.addEventHandler(WindowEvent.WINDOW_SHOWN, showingHandler);
		attachToSceneIfNeeded();
	}

	void uninstall() {
		var stage = qupath.getStage();
		stage.removeEventHandler(WindowEvent.WINDOW_SHOWN, showingHandler);
		Scene scene = stage.getScene();
		if (scene != null) {
			scene.removeEventFilter(KeyEvent.KEY_PRESSED, keyFilter);
		}
	}

	private void attachToSceneIfNeeded() {
		Scene scene = qupath.getStage().getScene();
		if (scene == null) {
			return;
		}
		scene.removeEventFilter(KeyEvent.KEY_PRESSED, keyFilter);
		scene.addEventFilter(KeyEvent.KEY_PRESSED, keyFilter);
	}

	private void handleKeyPressed(KeyEvent event) {
		if (event.isConsumed()) {
			return;
		}
		if (event.isShortcutDown() || event.isAltDown() || event.isControlDown() || event.isMetaDown()) {
			return;
		}
		if (isTypingInTextInput(event.getTarget())) {
			return;
		}
		KeyCode code = event.getCode();
		int index = digitIndex(code);
		if (index < 0) {
			return;
		}
		if (qupath.getViewer() == null || qupath.getViewer().getHierarchy() == null) {
			return;
		}
		PathObjectHierarchy hierarchy = qupath.getViewer().getHierarchy();
		List<PathObject> selected = new ArrayList<>(hierarchy.getSelectionModel().getSelectedObjects());
		if (selected.isEmpty()) {
			return;
		}
		String className = SAMAnnotClassShortcuts.getClassName(index);
		PathClass pathClass = PathClass.getInstance(className);
		if (!qupath.getAvailablePathClasses().contains(pathClass)) {
			qupath.getAvailablePathClasses().add(pathClass);
		}
		for (PathObject object : selected) {
			object.setPathClass(pathClass);
			Utils.syncAnnotationNameWithPathClass(object);
		}
		hierarchy.fireHierarchyChangedEvent(qupath);
		event.consume();
	}

	private static int digitIndex(KeyCode code) {
		return switch (code) {
			case DIGIT1, NUMPAD1 -> 0;
			case DIGIT2, NUMPAD2 -> 1;
			case DIGIT3, NUMPAD3 -> 2;
			case DIGIT4, NUMPAD4 -> 3;
			case DIGIT5, NUMPAD5 -> 4;
			case DIGIT6, NUMPAD6 -> 5;
			case DIGIT7, NUMPAD7 -> 6;
			case DIGIT8, NUMPAD8 -> 7;
			case DIGIT9, NUMPAD9 -> 8;
			default -> -1;
		};
	}

	private static boolean isTypingInTextInput(Object target) {
		Node node = target instanceof Node ? (Node) target : null;
		while (node != null) {
			if (node instanceof TextInputControl) {
				return true;
			}
			node = node.getParent();
		}
		return false;
	}

}
