package corpus;

public class FrequentConditionalStringVector {
	public final String index;
	public double[] vector;
	public int count;
	
	public FrequentConditionalStringVector(String i, double[] vector) {
		index = i;
		this.vector = vector;
	}
	
	@Override
	public int hashCode() {
		return index.hashCode();
	}
	
	@Override
	public boolean equals(Object obj) {
		if (obj == null)
            return false;
        if (obj == this)
            return true;
        if (!(obj instanceof FrequentConditionalStringVector))
            return false;
        FrequentConditionalStringVector other = (FrequentConditionalStringVector) obj;
        return index.equals(other.index);
	}
	
	@Override
	public String toString() {
		return index + " --> " + count;
	}
	
}
