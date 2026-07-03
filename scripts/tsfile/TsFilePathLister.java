import org.apache.tsfile.read.TsFileSequenceReader;
import org.apache.tsfile.read.common.Path;

import java.io.File;
import java.util.List;

/**
 * Small diagnostic utility: print measurement paths in one local TsFile.
 */
public class TsFilePathLister {
    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            throw new IllegalArgumentException("Usage: TsFilePathLister <file.tsfile> [max_paths]");
        }
        File file = new File(args[0]);
        int maxPaths = args.length >= 2 ? Integer.parseInt(args[1]) : 500;
        try (TsFileSequenceReader reader = new TsFileSequenceReader(file.getAbsolutePath())) {
            List<Path> paths = reader.getAllPaths();
            System.out.println("path_count=" + paths.size());
            int n = Math.min(maxPaths, paths.size());
            for (int i = 0; i < n; i++) {
                Path p = paths.get(i);
                System.out.println(i + "\t" + p.toString() + "\tmeasurement=" + p.getMeasurement());
            }
        }
    }
}
