package org.elephant.sam;

import java.util.List;

/**
 * Fixed path class names for digit shortcuts 1–9 (aligned with tem_sam3_viewer.py labels).
 */
public final class SAMAnnotClassShortcuts {

	private SAMAnnotClassShortcuts() {
	}

	private static final List<String> NAMES = List.of(
			"mitochondria_normal",
			"mitochondria_damaged",
			"nucleus",
			"condensed_chromatin",
			"autophagosome",
			"autolysosome",
			"apoptotic_body",
			"cell_membrane",
			"membrane_rupture");

	/**
	 * Class name for keys 1–9 at zero-based index {@code 0} … {@code 8}.
	 */
	public static String getClassName(int index) {
		return NAMES.get(index);
	}

	public static List<String> getClassNames() {
		return NAMES;
	}

	/**
	 * Multi-line legend for the Annotate tab (1-based keys).
	 */
	public static String formatLegend() {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < NAMES.size(); i++) {
			sb.append(i + 1).append(" → ").append(NAMES.get(i));
			if (i < NAMES.size() - 1) {
				sb.append('\n');
			}
		}
		return sb.toString();
	}

}
