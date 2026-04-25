package org.elephant.sam;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

import javafx.application.Platform;
import javafx.beans.value.ChangeListener;
import javafx.collections.ListChangeListener;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.viewer.QuPathViewer;
import qupath.lib.gui.viewer.ViewerManager;
import qupath.lib.images.ImageData;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.hierarchy.PathObjectHierarchy;
import qupath.lib.objects.hierarchy.events.PathObjectHierarchyEvent;
import qupath.lib.objects.hierarchy.events.PathObjectHierarchyListener;

/**
 * Keeps annotation names aligned with path class. Listens on every hierarchy that may be active,
 * handles classification and {@code CHANGE_OTHER} (property dialogs), defers work to the next
 * JavaFX frame so batched hierarchy updates finish first, and does not skip events solely because
 * {@link PathObjectHierarchyEvent#isChanging()} is true (that can drop class-assignment batches).
 */
final class SAMPathClassNameSync {

	private final QuPathGUI qupath;

	private final PathObjectHierarchyListener hierarchyListener = this::onHierarchyChanged;

	private final Set<PathObjectHierarchy> hierarchiesListening = new HashSet<>();

	private boolean usingViewerManager;

	private QuPathViewer trackedViewer;

	private final ChangeListener<ImageData<BufferedImage>> globalImageDataListener = (obs, o, n) -> syncHierarchyRegistrations();

	private final ChangeListener<ImageData<BufferedImage>> viewerImageDataListener = (obs, o, n) -> syncHierarchyRegistrations();

	private final ChangeListener<ImageData<BufferedImage>> perViewerImageDataListener = (obs, o, n) -> syncHierarchyRegistrations();

	private final ListChangeListener<QuPathViewer> viewersListListener = c -> {
		while (c.next()) {
			for (QuPathViewer removed : c.getRemoved()) {
				removed.imageDataProperty().removeListener(perViewerImageDataListener);
			}
			for (QuPathViewer added : c.getAddedSubList()) {
				added.imageDataProperty().addListener(perViewerImageDataListener);
			}
		}
		syncHierarchyRegistrations();
	};

	private final ChangeListener<QuPathViewer> viewerPropertyListener = (obs, oldViewer, newViewer) -> {
		if (oldViewer != null) {
			oldViewer.imageDataProperty().removeListener(viewerImageDataListener);
		}
		trackedViewer = newViewer;
		if (newViewer != null) {
			newViewer.imageDataProperty().addListener(viewerImageDataListener);
		}
		syncHierarchyRegistrations();
	};

	SAMPathClassNameSync(QuPathGUI qupath) {
		this.qupath = qupath;
	}

	void install() {
		qupath.imageDataProperty().addListener(globalImageDataListener);
		ViewerManager manager = getViewerManagerIfPresent();
		if (manager != null) {
			usingViewerManager = true;
			manager.getAllViewers().addListener(viewersListListener);
			for (QuPathViewer v : manager.getAllViewers()) {
				v.imageDataProperty().addListener(perViewerImageDataListener);
			}
		} else {
			usingViewerManager = false;
			qupath.viewerProperty().addListener(viewerPropertyListener);
			trackedViewer = qupath.getViewer();
			if (trackedViewer != null) {
				trackedViewer.imageDataProperty().addListener(viewerImageDataListener);
			}
		}
		syncHierarchyRegistrations();
	}

	void uninstall() {
		qupath.imageDataProperty().removeListener(globalImageDataListener);
		if (usingViewerManager) {
			ViewerManager manager = getViewerManagerIfPresent();
			if (manager != null) {
				manager.getAllViewers().removeListener(viewersListListener);
				for (QuPathViewer v : manager.getAllViewers()) {
					v.imageDataProperty().removeListener(perViewerImageDataListener);
				}
			}
		} else {
			qupath.viewerProperty().removeListener(viewerPropertyListener);
			if (trackedViewer != null) {
				trackedViewer.imageDataProperty().removeListener(viewerImageDataListener);
			}
			trackedViewer = null;
		}
		for (PathObjectHierarchy h : new ArrayList<>(hierarchiesListening)) {
			h.removeListener(hierarchyListener);
		}
		hierarchiesListening.clear();
	}

	private ViewerManager getViewerManagerIfPresent() {
		return qupath.getViewerManager();
	}

	private void syncHierarchyRegistrations() {
		for (PathObjectHierarchy h : new ArrayList<>(hierarchiesListening)) {
			h.removeListener(hierarchyListener);
		}
		hierarchiesListening.clear();
		registerHierarchy(imageDataHierarchy(qupath.getImageData()));
		if (usingViewerManager) {
			ViewerManager manager = getViewerManagerIfPresent();
			if (manager != null) {
				for (QuPathViewer v : manager.getAllViewers()) {
					registerHierarchy(imageDataHierarchy(v.getImageData()));
				}
			}
		} else if (qupath.getViewer() != null) {
			registerHierarchy(imageDataHierarchy(qupath.getViewer().getImageData()));
		}
	}

	private static PathObjectHierarchy imageDataHierarchy(ImageData<BufferedImage> data) {
		return data == null ? null : data.getHierarchy();
	}

	private void registerHierarchy(PathObjectHierarchy hierarchy) {
		if (hierarchy != null && hierarchiesListening.add(hierarchy)) {
			hierarchy.addListener(hierarchyListener);
		}
	}

	private void onHierarchyChanged(PathObjectHierarchyEvent event) {
		boolean isClassification = event.isObjectClassificationEvent();
		boolean isOther = event.getEventType() == PathObjectHierarchyEvent.HierarchyEventType.CHANGE_OTHER;
		if (!isClassification && !isOther) {
			return;
		}
		var changed = event.getChangedObjects();
		if (changed == null || changed.isEmpty()) {
			return;
		}
		// Defer so QuPath finishes batched hierarchy updates before we read path class / set names
		Platform.runLater(() -> {
			for (PathObject pathObject : changed) {
				if (isClassification) {
					Utils.syncAnnotationNameWithPathClass(pathObject);
				} else {
					Utils.syncAnnotationNameWithPathClassIfDefaultSamName(pathObject);
				}
			}
		});
	}
}
