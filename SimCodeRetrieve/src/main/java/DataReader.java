import com.alibaba.fastjson.JSONObject;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;


public class DataReader {
    public DataReader(){}

    public ArrayList<String> reader(String filePath) {
        try {
            File file = new File(filePath);
            if (file.isFile() && file.exists()) {
//                System.out.println("where is my file");
                InputStreamReader read = new InputStreamReader(new FileInputStream(file), StandardCharsets.UTF_8);
                BufferedReader bufferedReader = new BufferedReader(read);
                ArrayList<String> ret = new ArrayList<>();
                String lineTxt = null;
                while ((lineTxt =  bufferedReader.readLine())!=null){
                    ret.add(this.process(lineTxt));
//                    System.out.println(lineTxt);
                }
                return ret;
            }
        } catch (UnsupportedEncodingException | FileNotFoundException e) {
            System.out.println("Cannot find the file specified!");
            e.printStackTrace();
        } catch (IOException e) {
            System.out.println("Error reading file content!");
            e.printStackTrace();
        }
        return null;
    }

    private String process(String txtStr) {
        JSONObject json = JSONObject.parseObject(txtStr);
//        JSONArray jsonArray = new JSONArray();
//        json.forEach((key, value) -> {
//            System.out.println(key + ": "+ value);
//        });
        return json.getString("cleaned_comment");
    }

    public static void main(String[] args) {
        String path = "./data/test.valid";
        DataReader dataReader = new DataReader();
        ArrayList<String> l = dataReader.reader(path);
        assert l != null;
        int count = 0;
        for (String s : l){
            count ++;
            System.out.println(s);
        }
        System.out.println(count);
    }
}
