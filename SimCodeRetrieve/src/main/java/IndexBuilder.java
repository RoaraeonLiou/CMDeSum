import org.apache.lucene.document.Document;
import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.analysis.Analyzer;

import org.apache.lucene.document.TextField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.Field;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;

import java.io.*;

import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.lang.*;
import java.util.ArrayList;
import java.util.Iterator;

import org.apache.lucene.store.Directory;
import org.apache.lucene.store.NIOFSDirectory;

public class IndexBuilder {
    public IndexBuilder() {
    }

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

    /**
     * train_code   dir
     * 0            1
     */
//    public static void main(String[] args) throws Exception {
//        String[] new_args = {"../data/standard/train/train.token.code", "standard-corpus"};
//
//        for (Integer i = 0; i < new_args.length; ++i) {
//            System.out.print(i);
//            System.out.print(": ");
//            System.out.println(new_args[i]);
//        }
//        Path p = FileSystems.getDefault().getPath(new_args[1]);
//        Directory index = new NIOFSDirectory(p);
//        Analyzer analyzer = new WhitespaceAnalyzer();
//        IndexWriterConfig config = new IndexWriterConfig(analyzer);
//        IndexWriter w = new IndexWriter(index, config);
//
//        BufferedReader train_code = new BufferedReader(new FileReader(new_args[0]));
//
//        for (int i = 0; ; ++i) {
//            String thisLine = train_code.readLine();
//            if (thisLine == null) break;
//            Document d = new Document();
//            d.add(new TextField("code", escape(thisLine.trim()), Field.Store.YES));
//            d.add(new StoredField("No", i));
//            w.addDocument(d);
//        }
//        w.commit();
//    }
    public ArrayList<String> readFile(String filePath) {
        ArrayList<String> ret = new ArrayList<>();
        try {
            File file = new File(filePath);
            if (file.isFile() && file.exists()) {
                InputStreamReader read = new InputStreamReader(new FileInputStream(file), StandardCharsets.UTF_8);
                BufferedReader bufferedReader = new BufferedReader(read);
                String lineTxt = null;
                while ((lineTxt = bufferedReader.readLine()) != null) {
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

    public static Boolean deleteFile(String path) {
        File file = new File(path);
        File[] fileList = file.listFiles();
        if (fileList != null) {
            for (File subFile : fileList) {
                deleteFile(subFile.getAbsolutePath());
            }
        }
        return file.delete();
    }

    public void run(String inputFile, String outputDir) throws IOException {
        File file = new File(outputDir);
        if(file.exists() && !deleteFile(outputDir)){
            System.out.println("============Corpus is dirty!============");
        }
        Path p = FileSystems.getDefault().getPath(outputDir);
        Directory index = new NIOFSDirectory(p);
        Analyzer analyzer = new WhitespaceAnalyzer();
        IndexWriterConfig config = new IndexWriterConfig(analyzer);
        IndexWriter w = new IndexWriter(index, config);

        ArrayList<String> docList = this.readFile(inputFile);

        for (int i = 0; i < docList.size(); ++i) {
            Document d = new Document();
            d.add(new TextField("code", escape(docList.get(i).trim()), Field.Store.YES));
            d.add(new StoredField("No", i));
            w.addDocument(d);
        }
        w.commit();
        w.close();
    }

    public static void main(String[] args) throws Exception {
        deleteFile("./data/corpus");
//        String[] new_args = {"./data/tokenized_test.valid", "./data/standard-corpus"};
//
//        for (Integer i = 0; i < new_args.length; ++i) {
//            System.out.print(i);
//            System.out.print(": ");
//            System.out.println(new_args[i]);
//        }
//        Path p = FileSystems.getDefault().getPath(new_args[1]);
//        Directory index = new NIOFSDirectory(p);
//        Analyzer analyzer = new WhitespaceAnalyzer();
//        IndexWriterConfig config = new IndexWriterConfig(analyzer);
//        IndexWriter w = new IndexWriter(index, config);
//
//        DataReader dataReader = new DataReader();
//        ArrayList<String> commentList = dataReader.reader(new_args[0]);
//
//        for (int i = 0; i <commentList.size(); ++i) {
//            Document d = new Document();
//            d.add(new TextField("code", escape(commentList.get(i).trim()), Field.Store.YES));
//            d.add(new StoredField("No", i));
//            w.addDocument(d);
//        }
//        w.commit();
    }

}
