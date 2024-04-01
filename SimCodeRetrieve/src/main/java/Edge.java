public class Edge {
    public Edge(Integer from, Integer to){
        this.from = from;
        this.to = to;
    }

    public Edge(Integer from, Integer to, Integer cost){
        this.from = from;
        this.to = to;
        this.cost = cost;
    }

    public Integer from;
    public Integer to;
    public Integer cost;
}
