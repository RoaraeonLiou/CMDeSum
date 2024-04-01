import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

public class Blossom {
    public ArrayList<Integer> match;
    public ArrayList<Integer> aux;
    public ArrayList<Integer> label;
    public ArrayList<Integer> orig;
    public ArrayList<Integer> parent;
    public Queue<Integer> q;
    public Integer aux_time;
    public UndirectedGraph g;


    public Blossom(UndirectedGraph g) {
        this.g = g;
        // match
        this.match = new ArrayList<>();
        initArrayList(match, g.n, -1);
        // timestamp
        this.aux = new ArrayList<>();
        initArrayList(aux, g.n, -1);
        // [o] or [i]
        this.label = new ArrayList<>();
        initArrayList(label, g.n);
        // orig root
        this.orig = new ArrayList<>();
        initArrayList(orig, g.n);
        // parent node
        this.parent = new ArrayList<>();
        initArrayList(parent, g.n, -1);

        //
        this.q = new ArrayDeque<>();
        this.aux_time = -1;
    }

    public static <T> Boolean initArrayList(ArrayList<T> arrayList, int size, T initValue) {
        try {
            for (int i = 0; i < size; i++) {
                arrayList.add(initValue);
            }
            return true;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return false;
    }

    public static <T> Boolean initArrayList(ArrayList<T> arrayList, int size) {
        try {
            for (int i = 0; i < size; i++) {
                arrayList.add(null);
            }
            return true;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return false;
    }

    public static <T> Boolean fillArrayList(ArrayList<T> arrayList, T fillValue) {
        try {
            for (int i = 0; i < arrayList.size(); i++) {
                arrayList.set(i, fillValue);
            }
            return true;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return false;
    }

    public static Boolean iota(ArrayList<Integer> arrayList, Integer startValue) {
        try {
            for (int i = 0; i < arrayList.size(); i++) {
                arrayList.set(i, startValue);
                startValue++;
            }
            return true;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return false;
    }

    public Integer lca(Integer v, Integer u) {
        this.aux_time++;
        while (true) {
            if (v != -1) {
                if (this.aux.get(v).equals(this.aux_time)) {
                    return v;
                }
                this.aux.set(v, this.aux_time);
                if (this.match.get(v) == -1) {
                    v = -1;
                } else {
                    v = this.orig.get(this.parent.get(this.match.get(v)));
                }
            }
            // swap
            Integer temp = v;
            v = u;
            u = temp;
        }
    }

    public void blossom(Integer v, Integer u, Integer a) {
        while (!this.orig.get(v).equals(a)) {
            this.parent.set(v, u);
            u = this.match.get(v);
            if (this.label.get(u).equals(1)) {
                this.label.set(u, 0);
                this.q.add(u);
            }
            this.orig.set(v, a);
            this.orig.set(u, a);
            v = this.parent.get(u);
        }
    }

    public void augment(Integer v) {
        while (v != -1) {
            Integer pv = this.parent.get(v);
            Integer nextPv = this.match.get(pv);
            match.set(v, pv);
            match.set(pv, v);
            v = nextPv;
        }
    }

    public Boolean bfs(Integer root) {
        fillArrayList(this.label, -1);
        iota(this.orig, 0);
        while (!this.q.isEmpty()) {
            this.q.remove();
        }
        this.q.add(root);

        this.label.set(root, 0);
        while (!q.isEmpty()) {
            int v = q.peek();
            q.remove();
            for (int id : this.g.g.get(v)) {
                Edge e = this.g.edges.get(id);
                int u = e.from ^ e.to ^ v;
                if (label.get(u) == -1) {
                    label.set(u, 1);
                    parent.set(u, v);
                    if (match.get(u) == -1) {
                        augment(u);
                        return true;
                    }
                    label.set(match.get(u), 0);
                    q.add(match.get(u));
                    continue;
                } else if (label.get(u) == 0 && !orig.get(v).equals(orig.get(u))) {
                    int a = lca(orig.get(v), orig.get(u));
                    blossom(u, v, a);
                    blossom(v, u, a);
                }
            }
        }
        return false;
    }

    public void greedy() {
        ArrayList<Integer> order = new ArrayList<>();
        initArrayList(order, this.g.n);

        iota(order, 0);
        Collections.shuffle(order);

        for (Integer i : order) {
            if (this.match.get(i) == -1) {
                for (Integer id : this.g.g.get(i)) {
                    Edge e = this.g.edges.get(id);
                    int to = e.from ^ e.to ^ i;
                    if (this.match.get(to) == -1) {
                        this.match.set(i, to);
                        this.match.set(to, i);
                        break;
                    }
                }
            }
        }
    }

    public ArrayList<Integer> findMaxUnweightedMatching() {
        this.greedy();
        for (int i = 0; i < this.g.n; i++) {
            if (this.match.get(i) == -1) {
                this.bfs(i);
            }
        }
        return this.match;
    }

    public static ArrayList<Integer> readFile(String filePath) {
        try {
            File file = new File(filePath);
            if (file.isFile() && file.exists()) {
                InputStreamReader read = new InputStreamReader(new FileInputStream(file), StandardCharsets.UTF_8);
                BufferedReader bufferedReader = new BufferedReader(read);
                ArrayList<Integer> ret = new ArrayList<>();
                String lineTxt = null;
                while ((lineTxt = bufferedReader.readLine()) != null) {
                    Integer to = Integer.parseInt(lineTxt);
//                    ret.add(Map.of(from, to));
                    ret.add(to);
                }
                return ret;
            }
        } catch (UnsupportedEncodingException | FileNotFoundException e) {
            System.out.println("Cannot find the file specified!");
            e.printStackTrace();
        } catch (IOException e) {
            System.out.println("Error reading file content!");
            e.printStackTrace();
        } catch (NumberFormatException e) {
            System.out.println("Error parsing file content!");
            e.printStackTrace();
        }
        return null;
    }

    public Boolean writeFile(String filePath) {
        try {
            PrintWriter printWriter = new PrintWriter(filePath, "UTF-8");
            for (Integer integer : this.match) {
                printWriter.println(integer);
            }
            printWriter.close();
            return true;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return false;
    }

    public Integer count_pair() {
        int count = 0;
        for (Integer integer : this.match) {
            if (integer != -1) {
                count++;
            }
        }
        return count / 2 ;
    }

    public static void main(String[] args) {
        String inputFilePath = "./data/output/tokenized_test.valid.res";
        String outputFilePath = "./data/output/pair.res";
        ArrayList<Integer> edges = readFile(inputFilePath);
        System.out.println(edges);
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
}
