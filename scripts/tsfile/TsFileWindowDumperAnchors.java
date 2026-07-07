import org.apache.tsfile.read.TsFileReader;
import org.apache.tsfile.read.TsFileSequenceReader;
import org.apache.tsfile.read.common.Field;
import org.apache.tsfile.read.common.Path;
import org.apache.tsfile.read.common.RowRecord;
import org.apache.tsfile.read.expression.QueryExpression;
import org.apache.tsfile.read.query.dataset.QueryDataSet;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Convert TsFile samples with dataset-specific phase-transition anchor windows.
 *
 * Modes:
 *   standard_shift             - original 2->3 window, pre=30/post=70, with --shift
 *   standard_320321_anchors    - standard 16 features + anchors from datasetall_tsfile/320321gongkuang.py
 *   dataset13_anchors          - dataset13 native features + anchors from datasetall_tsfile/build_dataset15_1.py
 *   dataset14_anchors          - dataset14 native features + anchors from datasetall_tsfile/320321gongkuang.py
 *   standard_phase_start80     - standard 16 features + phase 0..12 starts, 80 rows each
 *   dataset13_phase_start80    - dataset13 native features + phase 0..12 starts, 80 rows each
 *   dataset14_phase_start80    - dataset14 native features + phase 0..12 starts, 80 rows each
 */
public class TsFileWindowDumperAnchors {
    private static final String PHASE_MEASUREMENT = "flight_phase";

    private static final String[] STANDARD_FEATURE_NAMES = new String[]{
            "N21", "N22", "BMPS1", "BMPS2",
            "PRECOOL_PRESS1", "PRECOOL_PRESS2",
            "PRV_ENG1_R", "PRV_ENG2_R",
            "HPV_ENG1_R", "HPV_ENG2_R",
            "PRECOOL_TEMP1", "PRECOOL_TEMP2",
            "PACK1_RAM_I_DR", "PACK1_RAM_O_DR",
            "PACK2_RAM_I_DR", "PACK2_RAM_O_DR"
    };
    private static final String[] STANDARD_MEASUREMENTS = new String[]{
            "n21", "n22", "bmps1", "bmps2",
            "precool_press1", "precool_press2",
            "prv_eng1_r", "prv_eng2_r",
            "hpv_eng1_r", "hpv_eng2_r",
            "precool_temp1", "precool_temp2",
            "pack1_ram_i_dr", "pack1_ram_o_dr",
            "pack2_ram_i_dr", "pack2_ram_o_dr"
    };

    private static final String[] DATASET13_FEATURES = new String[]{
            "caslcac1outempmp_01",
            "casrcac2outempmp_01",
            "casrcac2surgemp_c_01",
            "caslcac2rpmpm_ctl_01",
            "caslcac1rpmpm_ctl_01",
            "casrcac1oupresmp_01",
            "casrcac1rpmpm_ctl_01",
            "casrcac2rpmpm_ctl_01",
            "caslcac1oupresmp_01",
            "caslcac2outempmp_01",
            "casrcac1outempmp_01",
            "caslcac2oupresmp_01",
            "casrcac1surgemp_c_01",
            "casrpackcotempmp_01",
            "alt_std",
            "caslcac2surgemp_c_01",
            "casrcac2oupresmp_01",
            "ias",
            "caslcac1surgemp_c_01",
            "caslpackcotempmp_01"
    };

    private static final String[] DATASET14_FEATURES = new String[]{
            "n21", "n22",
            "e60031500l", "e60031500r",
            "e60041500l", "e60041500r",
            "e60091515l", "e60091515r"
    };

    private static class Anchor {
        final int from;
        final int to;
        final int pre;
        final int post;

        Anchor(int from, int to, int pre, int post) {
            this.from = from;
            this.to = to;
            this.pre = pre;
            this.post = post;
        }

        int length() {
            return pre + post;
        }

        String label() {
            return from + "->" + to;
        }
    }

    private static class Segment {
        final int anchor;
        final int start;
        final int end;

        Segment(int anchor, int start, int end) {
            this.anchor = anchor;
            this.start = start;
            this.end = end;
        }
    }

    private static class Sample {
        final int label;
        final String path;
        final String source;

        Sample(int label, String path, String source) {
            this.label = label;
            this.path = path;
            this.source = source;
        }
    }

    private static class Converted {
        final float[][] x;
        final float[] mask;
        int rows = 0;
        int firstTransition = -1;
        int validRows = 0;
        int missingFeatures = 0;
        boolean skip = false;
        String status = "OK";
        String message = "";

        Converted(int seqLen, int featureCount) {
            this.x = new float[seqLen][featureCount];
            this.mask = new float[seqLen];
        }
    }

    public static void main(String[] args) throws Exception {
        Map<String, String> opts = parseArgs(args);
        String manifestPath = require(opts, "--manifest");
        String outDir = require(opts, "--out");
        String mode = opts.getOrDefault("--mode", "standard_shift");
        int shift = Integer.parseInt(opts.getOrDefault("--shift", "-80"));
        boolean skipErrors = Boolean.parseBoolean(opts.getOrDefault("--skip_errors", "true"));

        String[] featureNames = featureNamesForMode(mode);
        String[] measurements = measurementsForMode(mode);
        Anchor[] anchors = anchorsForMode(mode);
        boolean strictAnchors = isStrictAnchorMode(mode);
        boolean phaseStartMode = isPhaseStartMode(mode);
        int seqLen = totalLength(anchors);
        int featureCount = featureNames.length;

        List<Sample> samples = readManifest(manifestPath);
        File out = new File(outDir);
        if (!out.exists() && !out.mkdirs()) {
            throw new IOException("Failed to create output dir: " + outDir);
        }

        try (DataOutputStream xOut = new DataOutputStream(new FileOutputStream(new File(out, "x.bin")));
             DataOutputStream maskOut = new DataOutputStream(new FileOutputStream(new File(out, "mask.bin")));
             DataOutputStream labelOut = new DataOutputStream(new FileOutputStream(new File(out, "labels.bin")));
             BufferedWriter stats = new BufferedWriter(new FileWriter(new File(out, "stats.tsv"), StandardCharsets.UTF_8))) {

            stats.write("idx\twritten_idx\tlabel\trows\tfirst_transition\tvalid_rows\tmissing_features\tstatus\tsource\tmessage\n");
            int written = 0;
            for (int i = 0; i < samples.size(); i++) {
                Sample sample = samples.get(i);
                Converted converted;
                try {
                    converted = convertOne(sample.path, mode, featureNames, measurements, anchors, seqLen,
                            featureCount, shift, strictAnchors, phaseStartMode);
                } catch (Throwable t) {
                    converted = new Converted(seqLen, featureCount);
                    converted.status = "ERROR";
                    converted.skip = true;
                    converted.message = t.getClass().getSimpleName() + ": " + t.getMessage();
                }

                if (skipErrors && converted.skip) {
                    writeStats(stats, i, -1, sample, converted);
                    continue;
                }

                for (int r = 0; r < seqLen; r++) {
                    for (int c = 0; c < featureCount; c++) {
                        xOut.writeFloat(converted.x[r][c]);
                    }
                }
                for (int r = 0; r < seqLen; r++) {
                    maskOut.writeFloat(converted.mask[r]);
                }
                labelOut.writeInt(sample.label);

                writeStats(stats, i, written, sample, converted);
                written++;
                if ((i + 1) % 500 == 0) {
                    System.err.println("converted " + (i + 1) + " / " + samples.size() + ", written " + written);
                }
            }
            writeMeta(out, written, seqLen, featureCount, shift, mode, featureNames, anchors);
        }
    }

    private static void writeStats(BufferedWriter stats, int idx, int writtenIdx, Sample sample,
                                   Converted converted) throws IOException {
        stats.write(idx + "\t" + writtenIdx + "\t" + sample.label + "\t" + converted.rows + "\t"
                + converted.firstTransition + "\t" + converted.validRows + "\t" + converted.missingFeatures
                + "\t" + converted.status + "\t" + sample.source + "\t"
                + sanitize(converted.message) + "\n");
    }

    private static Converted convertOne(String filePath, String mode, String[] featureNames, String[] measurements,
                                        Anchor[] anchors, int seqLen, int featureCount, int shift,
                                        boolean strictAnchors, boolean phaseStartMode) throws Exception {
        Converted out = new Converted(seqLen, featureCount);
        File file = new File(filePath);

        try (TsFileSequenceReader seqReader = new TsFileSequenceReader(file.getAbsolutePath());
             TsFileReader reader = new TsFileReader(seqReader)) {
            Map<String, Path> byMeasurement = new HashMap<>();
            for (Path p : seqReader.getAllPaths()) {
                String measurement = p.getMeasurement();
                if (measurement != null && !measurement.isEmpty()) {
                    byMeasurement.put(measurement.toLowerCase(Locale.ROOT), p);
                }
            }

            List<Path> selected = new ArrayList<>();
            selected.add(byMeasurement.get(PHASE_MEASUREMENT));
            if (selected.get(0) == null) {
                throw new IOException("missing measurement: " + PHASE_MEASUREMENT);
            }

            int[] selectedFeatureIndex = new int[featureCount];
            for (int i = 0; i < featureCount; i++) {
                selectedFeatureIndex[i] = -1;
                Path p = byMeasurement.get(measurements[i].toLowerCase(Locale.ROOT));
                if (p == null) {
                    out.missingFeatures++;
                } else {
                    selectedFeatureIndex[i] = selected.size();
                    selected.add(p);
                }
            }

            QueryDataSet dataSet = reader.query(QueryExpression.create(selected, null));
            List<float[]> rows = new ArrayList<>();
            List<Integer> phases = new ArrayList<>();
            while (dataSet.hasNext()) {
                RowRecord rec = dataSet.next();
                List<Field> fields = rec.getFields();
                int phase = Math.round(fieldToFloat(fields.size() > 0 ? fields.get(0) : null));
                float[] row = new float[featureCount];
                for (int c = 0; c < featureCount; c++) {
                    int fieldIdx = selectedFeatureIndex[c];
                    row[c] = fieldIdx < 0 || fieldIdx >= fields.size() ? 0.0f : fieldToFloat(fields.get(fieldIdx));
                    if (Float.isNaN(row[c]) || Float.isInfinite(row[c])) {
                        row[c] = 0.0f;
                    }
                }
                phases.add(phase);
                rows.add(row);
            }
            out.rows = rows.size();

            fillSingleZeroIfPresent(rows, featureNames, "N21");
            fillSingleZeroIfPresent(rows, featureNames, "N22");
            fillSingleZeroIfPresent(rows, featureNames, "n21");
            fillSingleZeroIfPresent(rows, featureNames, "n22");

            if (phaseStartMode) {
                copyPhaseStartWindows(out, rows, phases, anchors);
            } else if (strictAnchors) {
                copyStrictAnchorWindows(out, rows, phases, anchors);
            } else {
                copyPaddedShiftWindow(out, rows, phases, anchors[0], shift);
            }
            if (!out.skip) {
                instanceNorm(out.x, out.mask, featureCount);
            }
            return out;
        }
    }

    private static void copyPaddedShiftWindow(Converted out, List<float[]> rows, List<Integer> phases,
                                             Anchor anchor, int shift) {
        int transition = findTransition(phases, anchor.from, anchor.to);
        out.firstTransition = transition;
        if (transition < 0) {
            out.status = "NO_TRANSITION";
            return;
        }
        int center = transition + shift;
        int start = center - anchor.pre;
        int end = center + anchor.post;
        int srcStart = Math.max(start, 0);
        int srcEnd = Math.min(end, rows.size());
        if (srcEnd > srcStart) {
            int dst = srcStart - start;
            for (int i = srcStart; i < srcEnd; i++, dst++) {
                System.arraycopy(rows.get(i), 0, out.x[dst], 0, out.x[dst].length);
                out.mask[dst] = 1.0f;
                out.validRows++;
            }
        }
    }

    private static void copyStrictAnchorWindows(Converted out, List<float[]> rows, List<Integer> phases,
                                                Anchor[] anchors) {
        List<Segment> segments = new ArrayList<>();
        for (Anchor anchor : anchors) {
            int idx = findTransitionWithFallback(phases, anchor);
            if (out.firstTransition < 0) {
                out.firstTransition = idx;
            }
            if (idx < 0) {
                out.status = "MISSING_TRANSITION";
                out.message = anchor.label();
                out.skip = true;
                return;
            }
            if (idx < anchor.pre) {
                out.status = "PRE_SHORT";
                out.message = anchor.label() + " anchor=" + idx + " pre=" + anchor.pre;
                out.skip = true;
                return;
            }
            if (rows.size() - idx < anchor.post) {
                out.status = "POST_SHORT";
                out.message = anchor.label() + " after=" + (rows.size() - idx) + " post=" + anchor.post;
                out.skip = true;
                return;
            }
            segments.add(new Segment(idx, idx - anchor.pre, idx + anchor.post));
        }

        segments.sort((a, b) -> Integer.compare(a.anchor, b.anchor));
        int dst = 0;
        for (Segment segment : segments) {
            for (int i = segment.start; i < segment.end; i++, dst++) {
                System.arraycopy(rows.get(i), 0, out.x[dst], 0, out.x[dst].length);
                out.mask[dst] = 1.0f;
                out.validRows++;
            }
        }
    }

    private static void copyPhaseStartWindows(Converted out, List<float[]> rows, List<Integer> phases,
                                              Anchor[] anchors) {
        boolean partial = false;
        StringBuilder messages = new StringBuilder();
        int dstBase = 0;
        for (Anchor anchor : anchors) {
            int idx = findPhaseStart(phases, anchor.from);
            if (out.firstTransition < 0) {
                out.firstTransition = idx;
            }
            if (idx < 0) {
                partial = true;
                appendMessage(messages, "missing_phase=" + anchor.from);
                dstBase += anchor.length();
                continue;
            }
            int available = Math.max(0, Math.min(anchor.post, rows.size() - idx));
            if (available < anchor.post) {
                partial = true;
                appendMessage(messages, "phase=" + anchor.from + " after=" + available
                        + " post=" + anchor.post);
            }
            for (int k = 0; k < available; k++) {
                int dst = dstBase + k;
                System.arraycopy(rows.get(idx + k), 0, out.x[dst], 0, out.x[dst].length);
                out.mask[dst] = 1.0f;
                out.validRows++;
            }
            dstBase += anchor.length();
        }
        if (partial) {
            out.status = out.validRows > 0 ? "PARTIAL_PHASE_START" : "NO_VALID_PHASE_START";
            out.message = messages.toString();
        }
    }

    private static void appendMessage(StringBuilder sb, String message) {
        if (sb.length() > 0) {
            sb.append(";");
        }
        sb.append(message);
    }

    private static float fieldToFloat(Field field) {
        if (field == null || field.getDataType() == null) {
            return 0.0f;
        }
        Object value = field.getObjectValue(field.getDataType());
        if (value == null) {
            return 0.0f;
        }
        if (value instanceof Number) {
            return ((Number) value).floatValue();
        }
        if (value instanceof Boolean) {
            return ((Boolean) value) ? 1.0f : 0.0f;
        }
        try {
            return Float.parseFloat(value.toString());
        } catch (Exception e) {
            return 0.0f;
        }
    }

    private static int findTransition(List<Integer> phases, int from, int to) {
        for (int i = 1; i < phases.size(); i++) {
            if (phases.get(i - 1) == from && phases.get(i) == to) {
                return i;
            }
        }
        return -1;
    }

    private static int findPhaseStart(List<Integer> phases, int phase) {
        for (int i = 0; i < phases.size(); i++) {
            if (phases.get(i) == phase) {
                return i;
            }
        }
        return -1;
    }

    private static int findTransitionWithFallback(List<Integer> phases, Anchor anchor) {
        int idx = findTransition(phases, anchor.from, anchor.to);
        if (idx >= 0) {
            return idx;
        }
        if (anchor.from == 9 && anchor.to == 11) {
            return findTransition(phases, 10, 11);
        }
        return -1;
    }

    private static void fillSingleZeroIfPresent(List<float[]> rows, String[] featureNames, String name) {
        for (int c = 0; c < featureNames.length; c++) {
            if (featureNames[c].equalsIgnoreCase(name)) {
                fillSingleZero(rows, c);
            }
        }
    }

    private static void fillSingleZero(List<float[]> rows, int col) {
        for (int i = 1; i < rows.size() - 1; i++) {
            float prev = rows.get(i - 1)[col];
            float cur = rows.get(i)[col];
            float next = rows.get(i + 1)[col];
            if (cur == 0.0f && prev != 0.0f && next != 0.0f) {
                rows.get(i)[col] = (prev + next) / 2.0f;
            }
        }
    }

    private static void instanceNorm(float[][] x, float[] mask, int featureCount) {
        int valid = 0;
        for (float v : mask) {
            if (v > 0.0f) {
                valid++;
            }
        }
        if (valid <= 1) {
            for (int r = 0; r < x.length; r++) {
                if (mask[r] <= 0.0f) {
                    for (int c = 0; c < featureCount; c++) {
                        x[r][c] = 0.0f;
                    }
                }
            }
            return;
        }

        float[] mean = new float[featureCount];
        for (int r = 0; r < x.length; r++) {
            if (mask[r] > 0.0f) {
                for (int c = 0; c < featureCount; c++) {
                    mean[c] += x[r][c];
                }
            }
        }
        for (int c = 0; c < featureCount; c++) {
            mean[c] /= valid;
        }

        float[] var = new float[featureCount];
        for (int r = 0; r < x.length; r++) {
            if (mask[r] > 0.0f) {
                for (int c = 0; c < featureCount; c++) {
                    float d = x[r][c] - mean[c];
                    var[c] += d * d;
                }
            }
        }
        for (int c = 0; c < featureCount; c++) {
            var[c] = (float) Math.sqrt(var[c] / valid) + 1e-5f;
        }

        for (int r = 0; r < x.length; r++) {
            if (mask[r] > 0.0f) {
                for (int c = 0; c < featureCount; c++) {
                    x[r][c] = (x[r][c] - mean[c]) / var[c];
                }
            } else {
                for (int c = 0; c < featureCount; c++) {
                    x[r][c] = 0.0f;
                }
            }
        }
    }

    private static String[] featureNamesForMode(String mode) {
        if ("standard_shift".equals(mode)
                || "standard_320321_anchors".equals(mode)
                || "standard_phase_start80".equals(mode)) {
            return STANDARD_FEATURE_NAMES;
        }
        if ("dataset13_anchors".equals(mode) || "dataset13_phase_start80".equals(mode)) {
            return DATASET13_FEATURES;
        }
        if ("dataset14_anchors".equals(mode) || "dataset14_phase_start80".equals(mode)) {
            return DATASET14_FEATURES;
        }
        throw new IllegalArgumentException("Unknown mode: " + mode);
    }

    private static String[] measurementsForMode(String mode) {
        if ("standard_shift".equals(mode)
                || "standard_320321_anchors".equals(mode)
                || "standard_phase_start80".equals(mode)) {
            return STANDARD_MEASUREMENTS;
        }
        return featureNamesForMode(mode);
    }

    private static Anchor[] anchorsForMode(String mode) {
        if ("standard_shift".equals(mode)) {
            return new Anchor[]{new Anchor(2, 3, 30, 70)};
        }
        if ("standard_320321_anchors".equals(mode)) {
            return anchors320321();
        }
        if ("dataset13_anchors".equals(mode)) {
            return new Anchor[]{
                    new Anchor(0, 1, 30, 100),
                    new Anchor(1, 2, 30, 80),
                    new Anchor(2, 3, 30, 80),
                    new Anchor(3, 4, 30, 80),
                    new Anchor(4, 5, 100, 500),
                    new Anchor(5, 6, 200, 200),
                    new Anchor(8, 9, 200, 300),
                    new Anchor(9, 10, 200, 300),
                    new Anchor(10, 11, 80, 80),
                    new Anchor(11, 12, 80, 80),
                    new Anchor(12, 13, 80, 60)
            };
        }
        if ("dataset14_anchors".equals(mode)) {
            return anchors320321();
        }
        if ("standard_phase_start80".equals(mode)
                || "dataset13_phase_start80".equals(mode)
                || "dataset14_phase_start80".equals(mode)) {
            return phaseStart80Anchors();
        }
        throw new IllegalArgumentException("Unknown mode: " + mode);
    }

    private static boolean isStrictAnchorMode(String mode) {
        return "standard_320321_anchors".equals(mode)
                || "dataset13_anchors".equals(mode)
                || "dataset14_anchors".equals(mode);
    }

    private static boolean isPhaseStartMode(String mode) {
        return "standard_phase_start80".equals(mode)
                || "dataset13_phase_start80".equals(mode)
                || "dataset14_phase_start80".equals(mode);
    }

    private static Anchor[] anchors320321() {
        return new Anchor[]{
                new Anchor(0, 1, 30, 100),
                new Anchor(1, 2, 30, 80),
                new Anchor(2, 3, 30, 30),
                new Anchor(4, 5, 30, 500),
                new Anchor(5, 6, 200, 200),
                new Anchor(6, 8, 200, 300),
                new Anchor(8, 9, 200, 250),
                new Anchor(9, 11, 200, 80),
                new Anchor(11, 12, 5, 40),
                new Anchor(12, 13, 30, 200)
        };
    }

    private static Anchor[] phaseStart80Anchors() {
        Anchor[] anchors = new Anchor[13];
        for (int phase = 0; phase <= 12; phase++) {
            anchors[phase] = new Anchor(phase, phase, 0, 80);
        }
        return anchors;
    }

    private static int totalLength(Anchor[] anchors) {
        int total = 0;
        for (Anchor anchor : anchors) {
            total += anchor.length();
        }
        return total;
    }

    private static List<Sample> readManifest(String manifestPath) throws IOException {
        List<Sample> samples = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(manifestPath, StandardCharsets.UTF_8))) {
            String line;
            while ((line = br.readLine()) != null) {
                if (line.trim().isEmpty() || line.startsWith("#")) {
                    continue;
                }
                String[] parts = line.split("\t", 3);
                if (parts.length < 2) {
                    throw new IOException("Bad manifest line: " + line);
                }
                String source = parts.length >= 3 ? parts[2] : new File(parts[1]).getName();
                samples.add(new Sample(Integer.parseInt(parts[0]), parts[1], source));
            }
        }
        return samples;
    }

    private static void writeMeta(File out, int sampleCount, int seqLen, int featureCount, int shift,
                                  String mode, String[] featureNames, Anchor[] anchors) throws IOException {
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(new File(out, "meta.json"), StandardCharsets.UTF_8))) {
            bw.write("{\n");
            bw.write("  \"samples\": " + sampleCount + ",\n");
            bw.write("  \"seq_len\": " + seqLen + ",\n");
            bw.write("  \"feature_count\": " + featureCount + ",\n");
            bw.write("  \"phase_a_shift\": " + shift + ",\n");
            bw.write("  \"mode\": \"" + jsonEscape(mode) + "\",\n");
            bw.write("  \"feature_cols\": [");
            for (int i = 0; i < featureNames.length; i++) {
                if (i > 0) {
                    bw.write(", ");
                }
                bw.write("\"" + jsonEscape(featureNames[i]) + "\"");
            }
            bw.write("],\n");
            bw.write("  \"anchors\": [");
            for (int i = 0; i < anchors.length; i++) {
                if (i > 0) {
                    bw.write(", ");
                }
                Anchor a = anchors[i];
                bw.write("{\"from\": " + a.from + ", \"to\": " + a.to
                        + ", \"pre\": " + a.pre + ", \"post\": " + a.post + "}");
            }
            bw.write("]\n");
            bw.write("}\n");
        }
    }

    private static Map<String, String> parseArgs(String[] args) {
        Map<String, String> opts = new LinkedHashMap<>();
        for (int i = 0; i < args.length; i++) {
            String arg = args[i];
            if (!arg.startsWith("--")) {
                throw new IllegalArgumentException("Unexpected argument: " + arg);
            }
            if (i + 1 >= args.length) {
                throw new IllegalArgumentException("Missing value for " + arg);
            }
            opts.put(arg, args[++i]);
        }
        return opts;
    }

    private static String require(Map<String, String> opts, String key) {
        String value = opts.get(key);
        if (value == null || value.isEmpty()) {
            throw new IllegalArgumentException("Missing required option: " + key);
        }
        return value;
    }

    private static String sanitize(String value) {
        if (value == null) {
            return "";
        }
        return value.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ');
    }

    private static String jsonEscape(String value) {
        return value.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}
