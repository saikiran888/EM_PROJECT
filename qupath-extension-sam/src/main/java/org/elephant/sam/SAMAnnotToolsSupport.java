package org.elephant.sam;

import qupath.lib.gui.QuPathGUI;

/**
 * Installs optional SAM annotation keyboard shortcuts (digit keys 1–9).
 */
public final class SAMAnnotToolsSupport {

	private static SAMAnnotToolsSupport instance;

	private SAMKeyboardClassHandler keyboardHandler;

	private SAMPathClassNameSync pathClassNameSync;

	private SAMAnnotToolsSupport() {
	}

	/**
	 * Shared support object (extension installs once per application).
	 */
	public static synchronized SAMAnnotToolsSupport getInstance() {
		if (instance == null) {
			instance = new SAMAnnotToolsSupport();
		}
		return instance;
	}

	/**
	 * Install global digit shortcuts; safe to call once.
	 */
	public void installKeyboardHandler(QuPathGUI qupath) {
		if (keyboardHandler != null) {
			return;
		}
		keyboardHandler = new SAMKeyboardClassHandler(qupath);
		keyboardHandler.install();
	}

	/**
	 * When classification changes, rename objects that still use the default SAM name to the class label.
	 */
	public void installPathClassNameSync(QuPathGUI qupath) {
		if (pathClassNameSync != null) {
			return;
		}
		pathClassNameSync = new SAMPathClassNameSync(qupath);
		pathClassNameSync.install();
	}

	/**
	 * Remove global shortcuts (e.g. if extension uninstall API is added later).
	 */
	public void uninstallKeyboardHandler() {
		if (keyboardHandler != null) {
			keyboardHandler.uninstall();
			keyboardHandler = null;
		}
	}

	/**
	 * Detach classification/name sync (for symmetry with {@link #uninstallKeyboardHandler()}).
	 */
	public void uninstallPathClassNameSync() {
		if (pathClassNameSync != null) {
			pathClassNameSync.uninstall();
			pathClassNameSync = null;
		}
	}

}
