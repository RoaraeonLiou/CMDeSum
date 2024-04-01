import java.io.IOException;
import java.util.ArrayList;
import java.util.Map;

public class Run {
    public static void luceneProcess(String dataFilePath, String generatedCorpusDirPath, String outputResultFilePath) {
        try {
            IndexBuilder indexBuilder = new IndexBuilder();
            indexBuilder.run(dataFilePath, generatedCorpusDirPath);
            Searcher searcher = new Searcher();
            searcher.run(generatedCorpusDirPath, dataFilePath, outputResultFilePath);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void blossomProcess(String inputFilePath, String outputFilePath) {
        ArrayList<Integer> edges = Blossom.readFile(inputFilePath);
        assert edges != null;
        int edgeNums = edges.size();
        UndirectedGraph g = new UndirectedGraph(edgeNums);

        for (int i = 0; i < edgeNums; i++) {
            if (edges.get(i) != -1 && edges.get(i) != i) {
                g.add(i, edges.get(i));
            }
        }

        Blossom blossom = new Blossom(g);
        blossom.findMaxUnweightedMatching();
        Boolean flag = blossom.writeFile(outputFilePath);
        System.out.println("Generated " + blossom.count_pair().toString() + " pairs data.");
        System.out.println(flag ? "Output Success..." : "!!!Output Failed!!!");
    }

    //    public static void main(String[] args) {
//        String[] arguments = {"./data/tokenized_test.valid", "./data/corpus", "./data/output/tokenized_test.valid.res"};
//        IndexBuilder indexBuilder = new IndexBuilder();
//        indexBuilder.run(arguments[0], arguments[1]);
//        Searcher searcher = new Searcher();
//        searcher.run(arguments[1], arguments[0], arguments[2]);
//        System.out.println("Done!");
//    }
    public static void test() {
        Map<String, String> arg = Map.of(
                "dataFilePath", "./data/tokenized_test.valid",
                "generatedCorpusDirPath", "./data/corpus",
                "outputLuceneResultFilePath", "./data/output/tokenized_test.valid.res",
                "outputBlossomResultFilePath", "./data/output/pair.res"
        );
        // Step 1: Using Lucene build the index corpus and search similar code
        System.out.println("Now is the Lucene processes the data");
        luceneProcess(arg.get("dataFilePath"), arg.get("generatedCorpusDirPath"), arg.get("outputLuceneResultFilePath"));
        // Step 2: Using Blossom Algorithm to match the data pairs
        System.out.println("Now is the Blossom processes the data");
        blossomProcess(arg.get("outputLuceneResultFilePath"), arg.get("outputBlossomResultFilePath"));
    }
    public static void main(String[] args) throws Exception {
        /*
         * dataFilePath: The file path of output by python data process, only contains code data, and one line indicated one code.
         * generatedCorpusDirPath: The directory path of index corpus that Lucene will output.
         * outputLuceneResultFilePath: The file path where the Lucene will output its search result, which is also the input file path of the Blossom algorithm.
         * outputBlossomResultFilePath: The file path of Blossom algorithm output result.
         * */
        ArrayList<Map<String, String>> arguments = new ArrayList<>();
        ArrayList<String> dataSplit = new ArrayList<>();
        String base_path = "./data/csn/";
        String java = "java/";
        String python = "python/";
        String test = "test/";
        String valid = "valid/";
        String train = "train/";
        String tmp = "tmp/";
        String tokenFile = "code.token";
        String corpus = "corpus";
        String midFile = "sim.code.res";
        String resFile = "sim.pair.res";
        String javaTrain = base_path+java+train+tmp;
        String javaTest = base_path+java+test+tmp;
        String javaValid = base_path+java+valid+tmp;
        String pythonTrain = base_path+python+train+tmp;
        String pythonTest = base_path+python+test+tmp;
        String pythonValid = base_path+python+valid+tmp;
        // Train data
        dataSplit.add("java train");
        arguments.add(Map.of(
                "dataFilePath", javaTrain+tokenFile,
                "generatedCorpusDirPath", javaTrain+corpus,
                "outputLuceneResultFilePath", javaTrain+midFile,
                "outputBlossomResultFilePath", javaTrain+resFile
        ));
        // Test data
        dataSplit.add("java test");
        arguments.add(Map.of(
                "dataFilePath", javaTest+tokenFile,
                "generatedCorpusDirPath", javaTest+corpus,
                "outputLuceneResultFilePath", javaTest+midFile,
                "outputBlossomResultFilePath", javaTest+resFile
        ));
        // Valid data
        dataSplit.add("java valid");
        arguments.add(Map.of(
                "dataFilePath", javaValid+tokenFile,
                "generatedCorpusDirPath", javaValid+corpus,
                "outputLuceneResultFilePath", javaValid+midFile,
                "outputBlossomResultFilePath", javaValid+resFile
        ));
        // Python Train Data
        dataSplit.add("python train");
        arguments.add(Map.of(
                "dataFilePath", pythonTrain+tokenFile,
                "generatedCorpusDirPath", pythonTrain+corpus,
                "outputLuceneResultFilePath", pythonTrain+midFile,
                "outputBlossomResultFilePath", pythonTrain+resFile
        ));
        // Python Test Data
        dataSplit.add("python test");
        arguments.add(Map.of(
                "dataFilePath", pythonTest+tokenFile,
                "generatedCorpusDirPath", pythonTest+corpus,
                "outputLuceneResultFilePath", pythonTest+midFile,
                "outputBlossomResultFilePath", pythonTest+resFile
        ));
        // Python Valid Data
        dataSplit.add("python valid");
        arguments.add(Map.of(
                "dataFilePath", pythonValid+tokenFile,
                "generatedCorpusDirPath", pythonValid+corpus,
                "outputLuceneResultFilePath", pythonValid+midFile,
                "outputBlossomResultFilePath", pythonValid+resFile
        ));
        int index = 0;

        for (Map<String, String> arg : arguments) {
            if(index!=5){
                index++;
                continue;
            }
            System.out.println("**************************************************************");
            // Step 1: Using Lucene build the index corpus and search similar code
            System.out.println("Now is the Lucene processes the " + dataSplit.get(index)+" data");
            luceneProcess(arg.get("dataFilePath"), arg.get("generatedCorpusDirPath"), arg.get("outputLuceneResultFilePath"));
            // Step 2: Using Blossom Algorithm to match the data pairs
            System.out.println("Now is the Blossom processes the " + dataSplit.get(index)+" data");
            blossomProcess(arg.get("outputLuceneResultFilePath"), arg.get("outputBlossomResultFilePath"));
            index++;
//            System.out.println(arg);
        }
    }
}
