import org.apache.tsfile.read.TsFileReader;
import org.apache.tsfile.read.TsFileSequenceReader;
import org.apache.tsfile.read.common.Field;
import org.apache.tsfile.read.common.Path;
import org.apache.tsfile.read.common.RowRecord;
import org.apache.tsfile.read.expression.QueryExpression;
import org.apache.tsfile.read.query.dataset.QueryDataSet;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Diagnostic utility: print basic raw-value stats for selected measurements in one TsFile.
 */
public class TsFileMeasurementStats {
    private static class Stats {
        long count = 0;
        double sum = 0.0;
        double sumSq = 0.0;
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        Set<String> firstValues = new HashSet<>();

        void add(double value) {
            count++;
            sum += value;
            sumSq += value * value;
            min = Math.min(min, value);
            max = Math.max(max, value);
            if (firstValues.size() < 12) {
                firstValues.add(String.format(Locale.ROOT, "%.6g", value));
            }
        }

        double mean() {
            return count == 0 ? Double.NaN : sum / count;
        }

        double std() {
            if (count == 0) {
                return Double.NaN;
            }
            double mu = mean();
            return Math.sqrt(Math.max(0.0, sumSq / count - mu * mu));
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            throw new IllegalArgumentException("Usage: TsFileMeasurementStats <file.tsfile> <measurement> [measurement...]");
        }
        File file = new File(args[0]);
        try (TsFileSequenceReader seqReader = new TsFileSequenceReader(file.getAbsolutePath());
             TsFileReader reader = new TsFileReader(seqReader)) {
            Map<String, Path> byMeasurement = new HashMap<>();
            for (Path p : seqReader.getAllPaths()) {
                if (p.getMeasurement() != null) {
                    byMeasurement.put(p.getMeasurement().toLowerCase(Locale.ROOT), p);
                }
            }
            List<Path> selected = new ArrayList<>();
            List<String> selectedNames = new ArrayList<>();
            for (int i = 1; i < args.length; i++) {
                String name = args[i].toLowerCase(Locale.ROOT);
                Path p = byMeasurement.get(name);
                System.out.println("requested=" + args[i] + "\tfound=" + (p != null ? p.toString() : "NO"));
                if (p != null) {
                    selected.add(p);
                    selectedNames.add(args[i]);
                }
            }
            if (selected.isEmpty()) {
                return;
            }
            Stats[] stats = new Stats[selected.size()];
            for (int i = 0; i < stats.length; i++) {
                stats[i] = new Stats();
            }
            QueryDataSet dataSet = reader.query(QueryExpression.create(selected, null));
            while (dataSet.hasNext()) {
                RowRecord rec = dataSet.next();
                List<Field> fields = rec.getFields();
                for (int i = 0; i < selected.size(); i++) {
                    if (i >= fields.size() || fields.get(i) == null || fields.get(i).getDataType() == null) {
                        continue;
                    }
                    Object obj = fields.get(i).getObjectValue(fields.get(i).getDataType());
                    if (obj instanceof Number) {
                        stats[i].add(((Number) obj).doubleValue());
                    } else if (obj instanceof Boolean) {
                        stats[i].add(((Boolean) obj) ? 1.0 : 0.0);
                    } else {
                        try {
                            stats[i].add(Double.parseDouble(String.valueOf(obj)));
                        } catch (NumberFormatException ignored) {
                        }
                    }
                }
            }
            for (int i = 0; i < selected.size(); i++) {
                Stats s = stats[i];
                System.out.println(selectedNames.get(i)
                        + "\tcount=" + s.count
                        + "\tmin=" + s.min
                        + "\tmax=" + s.max
                        + "\tmean=" + s.mean()
                        + "\tstd=" + s.std()
                        + "\tfirst_values=" + s.firstValues);
            }
        }
    }
}
