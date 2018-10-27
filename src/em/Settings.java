package em;

public class Settings {
    int nIter;
    int burnin;
    boolean verbose;
    public Settings(int nIter, int burnin, boolean verbose){
        this.nIter = nIter;
        this.burnin = burnin;
        this.verbose = verbose;
    }
}
