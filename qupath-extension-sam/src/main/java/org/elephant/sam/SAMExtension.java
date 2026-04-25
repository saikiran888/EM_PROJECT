package org.elephant.sam;

import org.controlsfx.control.action.Action;
import org.elephant.sam.commands.SAMMainCommand;
import qupath.lib.gui.actions.ActionTools;
import qupath.lib.common.Version;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.actions.annotations.ActionMenu;
import qupath.lib.gui.extensions.QuPathExtension;

/**
 * QuPath extension for Segment Anything Model (SAM).
 */
public class SAMExtension implements QuPathExtension {

	/**
	 * Get the description of the extension.
	 * 
	 * @return The description of the extension.
	 */
	public String getDescription() {
		return "Run Segment Anything Model (SAM). Packaged by Sai.";
	}

	/**
	 * Get the name of the extension.
	 * 
	 * @return The name of the extension.
	 */
	public String getName() {
		return "SegmentAnything";
	}

	public void installExtension(QuPathGUI qupath) {
		qupath.installActions(ActionTools.getAnnotatedActions(new SAMCommands(qupath)));
		SAMAnnotToolsSupport support = SAMAnnotToolsSupport.getInstance();
		support.installKeyboardHandler(qupath);
		support.installPathClassNameSync(qupath);
	}

	@ActionMenu("Extensions")
	public class SAMCommands {

		public final Action actionSAMCommand;

		/**
		 * Constructor.
		 * 
		 * @param qupath
		 *            The QuPath GUI.
		 */
		private SAMCommands(QuPathGUI qupath) {
			SAMMainCommand samCommand = new SAMMainCommand(qupath);
			actionSAMCommand = new Action("SAM", event -> samCommand.run());
		}

	}

	@Override
	public Version getQuPathVersion() {
		return Version.parse("0.7.0");
	}

}