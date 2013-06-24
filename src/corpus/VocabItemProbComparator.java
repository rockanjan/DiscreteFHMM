package corpus;

import java.util.Comparator;

public class VocabItemProbComparator implements Comparator<VocabItemProbability> {
	@Override
	public int compare(VocabItemProbability v1, VocabItemProbability v2) {
		if(v1.prob < v2.prob) {
			return -1;
		}
		if(v1.prob > v2.prob) {
			return 1;
		}
		return 0;
	}
	
}
