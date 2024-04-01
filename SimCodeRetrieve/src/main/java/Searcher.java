import org.apache.lucene.document.Document;
import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.analysis.Analyzer;

import org.apache.lucene.document.TextField;
import org.apache.lucene.document.Field;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;

import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;

import org.apache.lucene.search.IndexSearcher;

import java.io.*;

import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.stream.*;
import java.util.function.*;
import java.lang.*;
import java.util.Iterator;

import java.util.Collection;
import java.util.Collections;

import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.NIOFSDirectory;


public class Searcher {
    public Searcher(){}

    private static String escape(String a) {
        return a
            .replace("\\", "\\\\")
            .replace("+", "\\+")
            .replace("-", "\\-")
            .replace("&", "\\&")
            .replace("|", "\\|")
            .replace("!", "\\!")
            .replace("(", "\\(")
            .replace(")", "\\)")
            .replace("{", "\\{")
            .replace("}", "\\}")
            .replace("[", "\\[")
            .replace("]", "\\]")
            .replace("^", "\\^")
            .replace("\"", "\\\"")
            .replace("~", "\\~")
            .replace("*", "\\*")
            .replace("?", "\\?")
            .replace(":", "\\:")
            .replace("/", "\\/")
            .replace("OR", "aseORase")
            .replace("AND", "aseANDase")
            .replace("NOT", "aseNOTase");
    }

    public static ArrayList<String> readFile(String filePath){
        ArrayList<String> ret = new ArrayList<>();
        try {
            File file = new File(filePath);
            if (file.isFile() && file.exists()) {
                InputStreamReader read = new InputStreamReader(new FileInputStream(file), StandardCharsets.UTF_8);
                BufferedReader bufferedReader = new BufferedReader(read);
                String lineTxt = null;
                while ((lineTxt =  bufferedReader.readLine())!=null){
                    ret.add(lineTxt);
                }
            }
        } catch (UnsupportedEncodingException | FileNotFoundException e) {
            System.out.println("Cannot find the file specified!");
            e.printStackTrace();
        } catch (IOException e) {
            System.out.println("Error reading file content!");
            e.printStackTrace();
        }
        return ret;
    }
    
    private static Integer searching(String source, String res_exemplar, IndexSearcher searcher, QueryParser parser) throws Exception {
//        BufferedReader test_code = new BufferedReader(new FileReader(source));
        DataReader dataReader = new DataReader();
//        ArrayList<String> commentList = dataReader.reader(source);
        ArrayList<String> docList = readFile(source);
        PrintWriter exemplar_res = new PrintWriter(res_exemplar, "UTF-8");
        String x;

        for (String str : docList){
            try {
                String escaped = escape(str.trim());
                ScoreDoc[] D = searcher.search(parser.parse(escaped), 2).scoreDocs;
                Document d0 = searcher.doc(D[0].doc);
                if (!escaped.equals(d0.get("code"))) {
//                    System.out.println(d0.get("code"));
//                    System.out.println(escaped);
                    exemplar_res.println(d0.get("No"));
                } else {
//                    System.out.println(d0.get("code").equals(escaped));
//                    System.out.println(searcher.doc(D[1].doc).get("No"));
//                    System.out.println(searcher.doc(D[0].doc).get("No"));
//                    System.out.println(escaped);
                    exemplar_res.println(searcher.doc(D[1].doc).get("No"));
                }
            } catch (Exception e) {
                exemplar_res.println("-1");
            }
        }
        exemplar_res.close();
        return 1;
    }

//    public static int searchAndDelete(String doc, IndexSearcher searcher, QueryParser parser) throws ParseException, IOException {
//        String escaped = escape(doc);
//        ScoreDoc[] D = searcher.search(parser.parse(escaped), 2).scoreDocs;
//        Document d0 = searcher.doc(D[0].doc);
//        if (!escaped.equals(d0.get("code"))) {
//            exemplar_res.println(d0.get("No"));
//        }
//    }

    public void run(String indexCorpus, String inputFilePath, String outputFilePath) throws Exception {
        Path p = FileSystems.getDefault().getPath(indexCorpus);
        Directory index = new NIOFSDirectory(p);
        Analyzer analyzer = new WhitespaceAnalyzer();
        IndexWriterConfig config = new IndexWriterConfig(analyzer);
        config.setOpenMode(IndexWriterConfig.OpenMode.APPEND);
        IndexWriter w = new IndexWriter(index, config);

        IndexReader r = DirectoryReader.open(w);

        IndexSearcher searcher = new IndexSearcher(r);

        QueryParser parser = new QueryParser("code", analyzer);
        System.out.print("processing ");
        System.out.println(inputFilePath);
        System.out.println(searching(inputFilePath, outputFilePath, searcher, parser));
        r.close();
        w.close();
    }

    /**
     * dir     test_code   exemplar
     * 0       1           2
     */
    public static void main(String[] args) throws Exception {
//        String[] new_args = {"./data/standard-corpus", "./data/test2.valid", "./data/exemplar.example"};
//        for (Integer i = 0; i < args.length; ++i) {
//            System.out.print(i);
//            System.out.print(": ");
//            System.out.println(args[i]);
//        }
//        Path p = FileSystems.getDefault().getPath(new_args[0]);
//        Directory index = new NIOFSDirectory(p);
//        Analyzer analyzer = new WhitespaceAnalyzer();
//        IndexWriterConfig config = new IndexWriterConfig(analyzer);
//        config.setOpenMode(IndexWriterConfig.OpenMode.APPEND);
//        IndexWriter w = new IndexWriter(index, config);
//
//        IndexReader r = DirectoryReader.open(w);
//
//        IndexSearcher searcher = new IndexSearcher(r);
//
//        QueryParser parser = new QueryParser("code", analyzer);
//
//        for (int i = 1; i < new_args.length; i += 2) {
//            System.out.print("processing ");
//            System.out.println(new_args[i]);
//            System.out.println(searching(new_args[i], new_args[i+1], searcher, parser));
//        }
    }
}
