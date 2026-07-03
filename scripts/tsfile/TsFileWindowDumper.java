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
 * Convert local Apache TsFile flight samples into fixed QAR classification windows.
 *
 * Input manifest format, one sample per line:
 *   label<TAB>absolute_tsfile_path<TAB>source_name
 *
 * Output files under --out:
 *   x.bin        big-endian float32, shape [N, 100, 16]
 *   mask.bin     big-endian float32, shape [N, 100]
 *   labels.bin   big-endian int32, shape [N]
 *   stats.tsv    per-file conversion diagnostics
 *   meta.json    dimensions and feature names
 */
public class TsFileWindowDumper {
    private static final String[] FEATURE_NAMES = new String[]{
            "N21", "N22", "BMPS1", "BMPS2",
            "PRECOOL_PRESS1", "PRECOOL_PRESS2",
            "PRV_ENG1_R", "PRV_ENG2_R",
            "HPV_ENG1_R", "HPV_ENG2_R",
            "PRECOOL_TEMP1", "PRECOOL_TEMP2",
            "PACK1_RAM_I_DR", "PACK1_RAM_O_DR",
            "PACK2_RAM_I_DR", "PACK2_RAM_O_DR"
    };
    private static final String[] TS_MEASUREMENTS = new String[]{
            "n21", "n22", "bmps1", "bmps2",
            "precool_press1", "precool_press2",
            "prv_eng1_r", "prv_eng2_r",
            "hpv_eng1_r", "hpv_eng2_r",
            "precool_temp1", "precool_temp2",
            "pack1_ram_i_dr", "pack1_ram_o_dr",
            "pack2_ram_i_dr", "pack2_ram_o_dr"
    };
    private static final String PHASE_MEASUREMENT = "flight_phase";
    private static final int SEG_A_PRE = 30;
    private static final int SEG_A_POST = 70;
    private static final int SEQ_LEN = SEG_A_PRE + SEG_A_POST;
    private static final int FEATURE_COUNT = FEATURE_NAMES.length;

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
        final float[][] x = new float[SEQ_LEN][FEATURE_COUNT];
        final float[] mask = new float[SEQ_LEN];
        int rows = 0;
        int transition = -1;
        int validRows = 0;
        int missingFeatures = 0;
        String status = "OK";
        String message = "";
    }

    public static void main(String[] args) throws Exception {
        Map<String, String> opts = parseArgs(args);
        String manifestPath = require(opts, "--manifest");
        String outDir = require(opts, "--out");
        int shift = Integer.parseInt(opts.getOrDefault("--shift", "-80"));
        boolean skipErrors = Boolean.parseBoolean(opts.getOrDefault("--skip_errors", "true"));

        List<Sample> samples = readManifest(manifestPath);
        File out = new File(outDir);
        if (!out.exists() && !out.mkdirs()) {
            throw new IOException("Failed to create output dir: " + outDir);
        }

        try (DataOutputStream xOut = new DataOutputStream(new FileOutputStream(new File(out, "x.bin")));
             DataOutputStream maskOut = new DataOutputStream(new FileOutputStream(new File(out, "mask.bin")));
             DataOutputStream labelOut = new DataOutputStream(new FileOutputStream(new File(out, "labels.bin")));
             BufferedWriter stats = new BufferedWriter(new FileWriter(new File(out, "stats.tsv"), StandardCharsets.UTF_8))) {

            stats.write("idx\twritten_idx\tlabel\trows\ttransition\tvalid_rows\tmissing_features\tstatus\tsource\tmessage\n");
            int written = 0;
            for (int i = 0; i < samples.size(); i++) {
                Sample sample = samples.get(i);
                Converted converted;
                try {
                    converted = convertOne(sample.path, shift);
                } catch (Throwable t) {
                    converted = new Converted();
                    converted.status = "ERROR";
                    converted.message = t.getClass().getSimpleName() + ": " + t.getMessage();
                }

                if (skipErrors && "ERROR".equals(converted.status)) {
                    stats.write(i + "\t-1\t" + sample.label + "\t" + converted.rows + "\t" + converted.transition + "\t"
                            + converted.validRows + "\t" + converted.missingFeatures + "\t" + converted.status + "\t"
                            + sample.source + "\t" + sanitize(converted.message) + "\n");
                    continue;
                }

                for (int r = 0; r < SEQ_LEN; r++) {
                    for (int c = 0; c < FEATURE_COUNT; c++) {
                        xOut.writeFloat(converted.x[r][c]);
                    }
                }
                for (int r = 0; r < SEQ_LEN; r++) {
                    maskOut.writeFloat(converted.mask[r]);
                }
                labelOut.writeInt(sample.label);

                stats.write(i + "\t" + written + "\t" + sample.label + "\t" + converted.rows + "\t" + converted.transition + "\t"
                        + converted.validRows + "\t" + converted.missingFeatures + "\t" + converted.status + "\t"
                        + sample.source + "\t" + sanitize(converted.message) + "\n");
                written++;
                if ((i + 1) % 500 == 0) {
                    System.err.println("converted " + (i + 1) + " / " + samples.size() + ", written " + written);
                }
            }
            writeMeta(out, written, shift);
        }
    }

    private static Converted convertOne(String filePath, int shift) throws Exception {
        Converted out = new Converted();
        File file = new File(filePath);

        try (TsFileSequenceReader seqReader = new TsFileSequenceReader(file.getAbsolutePath());
             TsFileReader reader = new TsFileReader(seqReader)) {
            List<Path> allPaths = seqReader.getAllPaths();
            Map<String, Path> byMeasurement = new HashMap<>();
            for (Path p : allPaths) {
                String measurement = p.getMeasurement();
                if (measurement != null) {
                    byMeasurement.put(measurement.toLowerCase(Locale.ROOT), p);
                }
            }

            List<Path> selected = new ArrayList<>();
            selected.add(byMeasurement.get(PHASE_MEASUREMENT));
            if (selected.get(0) == null) {
                throw new IOException("missing measurement: " + PHASE_MEASUREMENT);
            }

            int[] selectedFeatureIndex = new int[FEATURE_COUNT];
            for (int i = 0; i < FEATURE_COUNT; i++) {
                selectedFeatureIndex[i] = -1;
                Path p = byMeasurement.get(TS_MEASUREMENTS[i]);
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
                float[] row = new float[FEATURE_COUNT];
                for (int c = 0; c < FEATURE_COUNT; c++) {
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
            fillSingleZero(rows, 0); // n21
            fillSingleZero(rows, 1); // n22

            int transition = findTransition(phases, 2, 3);
            out.transition = transition;
            if (transition < 0) {
                out.status = "NO_TRANSITION";
                return out;
            }
            int center = transition + shift;
            int start = center - SEG_A_PRE;
            int end = center + SEG_A_POST;
            int srcStart = Math.max(start, 0);
            int srcEnd = Math.min(end, rows.size());
            if (srcEnd > srcStart) {
                int dst = srcStart - start;
                for (int i = srcStart; i < srcEnd; i++, dst++) {
                    System.arraycopy(rows.get(i), 0, out.x[dst], 0, FEATURE_COUNT);
                    out.mask[dst] = 1.0f;
                    out.validRows++;
                }
            }
            instanceNorm(out.x, out.mask);
            return out;
        }
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

    private static void instanceNorm(float[][] x, float[] mask) {
        int valid = 0;
        for (float v : mask) {
            if (v > 0.0f) {
                valid++;
            }
        }
        if (valid <= 1) {
            for (int r = 0; r < x.length; r++) {
                if (mask[r] <= 0.0f) {
                    for (int c = 0; c < FEATURE_COUNT; c++) {
                        x[r][c] = 0.0f;
                    }
                }
            }
            return;
        }

        float[] mean = new float[FEATURE_COUNT];
        for (int r = 0; r < x.length; r++) {
            if (mask[r] > 0.0f) {
                for (int c = 0; c < FEATURE_COUNT; c++) {
                    mean[c] += x[r][c];
                }
            }
        }
        for (int c = 0; c < FEATURE_COUNT; c++) {
            mean[c] /= valid;
        }

        float[] var = new float[FEATURE_COUNT];
        for (int r = 0; r < x.length; r++) {
            if (mask[r] > 0.0f) {
                for (int c = 0; c < FEATURE_COUNT; c++) {
                    float d = x[r][c] - mean[c];
                    var[c] += d * d;
                }
            }
        }
        for (int c = 0; c < FEATURE_COUNT; c++) {
            var[c] = (float) Math.sqrt(var[c] / valid) + 1e-5f;
        }

        for (int r = 0; r < x.length; r++) {
            if (mask[r] > 0.0f) {
                for (int c = 0; c < FEATURE_COUNT; c++) {
                    x[r][c] = (x[r][c] - mean[c]) / var[c];
                }
            } else {
                for (int c = 0; c < FEATURE_COUNT; c++) {
                    x[r][c] = 0.0f;
                }
            }
        }
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

    private static void writeMeta(File out, int sampleCount, int shift) throws IOException {
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(new File(out, "meta.json"), StandardCharsets.UTF_8))) {
            bw.write("{\n");
            bw.write("  \"samples\": " + sampleCount + ",\n");
            bw.write("  \"seq_len\": " + SEQ_LEN + ",\n");
            bw.write("  \"feature_count\": " + FEATURE_COUNT + ",\n");
            bw.write("  \"phase_a_shift\": " + shift + ",\n");
            bw.write("  \"feature_cols\": [");
            for (int i = 0; i < FEATURE_NAMES.length; i++) {
                if (i > 0) {
                    bw.write(", ");
                }
                bw.write("\"" + FEATURE_NAMES[i] + "\"");
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
}
