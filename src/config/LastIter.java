package config;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;

public class LastIter {
	public static final String lastSavedIterFile = "iterFile.txt";

	public static void write(int iter) throws FileNotFoundException {
		PrintWriter pw = new PrintWriter(new File(lastSavedIterFile));
		pw.println(iter);
		pw.close();
	}
	
	public static int read() throws NumberFormatException, IOException {
		File f = new File(lastSavedIterFile);
		if(! f.exists()) {
			return -1;
		}
		BufferedReader br = new BufferedReader(new FileReader(f));
		Integer iter = Integer.parseInt(br.readLine().trim());
		br.close();
		return iter;
	}
}
