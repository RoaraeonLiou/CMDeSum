import java.util.ArrayList;

public class Graph {
    public ArrayList<Edge> edges;
    public ArrayList<ArrayList<Integer>> g;
    public Integer n;

    public Graph(int _n) {
        this.n = _n;
        this.g = new ArrayList<>(this.n);
        for(int i=0;i<this.n;i++){
            this.g.add(new ArrayList<>());
        }
        this.edges = new ArrayList<>();
    }
}
