public class UndirectedGraph extends Graph{
    public UndirectedGraph(int _n) {
        super(_n);
    }

    public int add(Integer from, Integer to){
        return this.add(from, to, 1);
    }

    public int add(Integer from, Integer to, int cost){
        assert (0 <= from && from < n && 0 <= to && to < n);
        int id = (int)edges.size();
        g.get(from).add(id);
        g.get(to).add(id);
        edges.add(new Edge(from, to, cost));
        return id;
    }
}
