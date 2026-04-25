package org.elephant.sam.ui;

import java.util.ArrayList;
import java.util.List;

import org.controlsfx.control.action.Action;
import org.elephant.sam.SAMAnnotClassShortcuts;
import org.elephant.sam.commands.SAMMainCommand;

import javafx.collections.FXCollections;
import javafx.geometry.Insets;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.Separator;
import javafx.scene.control.ToggleButton;
import javafx.scene.control.Tooltip;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Pane;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.actions.ActionTools;
import qupath.lib.gui.viewer.tools.PathTools;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.classes.PathClass;

/**
 * Manual drawing tools (reuse QuPath brush / polygon / move) and class assignment from a list.
 */
public class SAMAnnotToolsPane extends VBox {

	public SAMAnnotToolsPane(SAMMainCommand command) {
		super(8);
		setPadding(new Insets(8));

		Label title = new Label("Draw annotations");
		title.setStyle("-fx-font-weight: bold;");

		// Do not call setTooltip on these toggles: ActionTools binds tooltip from the PathTool Action.
		Action brush = command.getQuPath().getToolManager().getToolAction(PathTools.BRUSH);
		ToggleButton btnBrush = ActionTools.createToggleButtonWithGraphicOnly(brush);

		Action polygon = command.getQuPath().getToolManager().getToolAction(PathTools.POLYGON);
		ToggleButton btnPolygon = ActionTools.createToggleButtonWithGraphicOnly(polygon);

		Action move = command.getQuPath().getToolManager().getToolAction(PathTools.MOVE);
		ToggleButton btnMove = ActionTools.createToggleButtonWithGraphicOnly(move);

		HBox tools = new HBox(8, btnBrush, btnPolygon, btnMove);

		Button btnDelete = new Button("Delete selected");
		btnDelete.setMaxWidth(Double.MAX_VALUE);
		btnDelete.setTooltip(new Tooltip("Remove selected objects from the current hierarchy."));
		btnDelete.setOnAction(e -> deleteSelected(command));

		Separator sep = new Separator();

		Label classHeading = new Label("Class for selected contours");
		classHeading.setStyle("-fx-font-weight: bold;");
		classHeading.setTooltip(new Tooltip(
				"Select contour(s) in the viewer, then pick a class here. "
						+ "The classification and object name are both set. See the Help tab for digit shortcuts."));

		ComboBox<String> comboClass = new ComboBox<>(FXCollections.observableArrayList(SAMAnnotClassShortcuts.getClassNames()));
		comboClass.setMaxWidth(Double.MAX_VALUE);
		comboClass.setPromptText("Choose class…");
		comboClass.setTooltip(new Tooltip(
				"Select contour(s) in the image, then pick a class. Classification and name are updated."));
		comboClass.setOnAction(e -> {
			String name = comboClass.getValue();
			if (name == null || name.isBlank()) {
				return;
			}
			applyClassAndNameToSelected(command, name);
			comboClass.getSelectionModel().clearSelection();
			comboClass.setValue(null);
		});

		getChildren().addAll(title, tools, btnDelete, sep, classHeading, comboClass);

		Pane spacer = new Pane();
		VBox.setVgrow(spacer, Priority.ALWAYS);
		getChildren().add(spacer);
	}

	private static void applyClassAndNameToSelected(SAMMainCommand command, String className) {
		QuPathGUI qupath = command.getQuPath();
		var viewer = qupath.getViewer();
		if (viewer == null) {
			return;
		}
		var hierarchy = viewer.getHierarchy();
		List<PathObject> selected = new ArrayList<>(hierarchy.getSelectionModel().getSelectedObjects());
		if (selected.isEmpty()) {
			return;
		}
		PathClass pathClass = PathClass.getInstance(className);
		if (!qupath.getAvailablePathClasses().contains(pathClass)) {
			qupath.getAvailablePathClasses().add(pathClass);
		}
		for (PathObject obj : selected) {
			obj.setPathClass(pathClass);
			obj.setName(className);
		}
		hierarchy.fireHierarchyChangedEvent(command);
	}

	private static void deleteSelected(SAMMainCommand command) {
		var viewer = command.getQuPath().getViewer();
		if (viewer == null) {
			return;
		}
		var hierarchy = viewer.getHierarchy();
		List<PathObject> selected = new ArrayList<>(hierarchy.getSelectionModel().getSelectedObjects());
		if (selected.isEmpty()) {
			return;
		}
		hierarchy.removeObjects(selected, true);
		hierarchy.getSelectionModel().clearSelection();
		hierarchy.fireHierarchyChangedEvent(command);
	}

}
