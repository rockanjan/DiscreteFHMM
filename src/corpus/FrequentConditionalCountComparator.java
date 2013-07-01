package corpus;

import java.util.Comparator;

public class FrequentConditionalCountComparator implements Comparator<FrequentConditionalStringVector> {
	
	//for min-heap
	@Override
	public int compare(FrequentConditionalStringVector v1, FrequentConditionalStringVector v2) {
		if(v1.count < v2.count) {
			return -1;
		}
		if(v1.count > v2.count) {
			return 1;
		}
		return 0;
	}
	
}
