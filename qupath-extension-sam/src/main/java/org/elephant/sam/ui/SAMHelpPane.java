package org.elephant.sam.ui;

import org.elephant.sam.SAMAnnotClassShortcuts;

import javafx.geometry.Insets;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.Separator;
import javafx.scene.layout.VBox;
import javafx.scene.text.TextAlignment;

/**
 * Help text for the SAM window (shortcuts, viewer behaviour).
 */
public class SAMHelpPane extends ScrollPane {

	public SAMHelpPane() {
		setFitToWidth(true);

		Label headingKeys = new Label("Annotate tab — keyboard shortcuts");
		headingKeys.setStyle("-fx-font-weight: bold;");
		headingKeys.setWrapText(true);

		Label keysIntro = new Label(
				"With the main QuPath window focused (not typing in a text field), select one or more "
						+ "objects and press a digit key (1–9 or numpad). Modifier keys "
						+ "(Ctrl/Cmd/Alt/Meta) disable the shortcut so keys still work elsewhere.");
		keysIntro.setWrapText(true);

		Label keysLegend = new Label(SAMAnnotClassShortcuts.formatLegend());
		keysLegend.setWrapText(true);
		keysLegend.setStyle("-fx-font-family: monospace;");

		Separator sep1 = new Separator();

		Label headingNames = new Label("Annotation names on the image");
		headingNames.setStyle("-fx-font-weight: bold;");
		headingNames.setWrapText(true);

		Label namesBody = new Label(
				"QuPath draws object names with its built-in viewer overlay. This extension can update "
						+ "the stored name and classification, and can toggle “Display names” on the Prompt tab, "
						+ "but it cannot move labels to the centre or inside contours — that behaviour is fixed "
						+ "inside QuPath (HierarchyOverlay), not in extension code.");
		namesBody.setWrapText(true);
		namesBody.setTextAlignment(TextAlignment.LEFT);

		VBox box = new VBox(10,
				headingKeys,
				keysIntro,
				keysLegend,
				sep1,
				headingNames,
				namesBody);
		box.setPadding(new Insets(8));
		setContent(box);
	}

}
