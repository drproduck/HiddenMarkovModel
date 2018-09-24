package hmmWithSpacedCharacter;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

public class utilSpace {


    public static double[][][] trigram(int len, int skip, boolean normalized) throws Exception{
        int[] chars = textProcess("brown_nolines.txt", skip, len);
        double[][][] trigramFreq = new double[52][52][52];
        int a = chars[0];
        int b = chars[1];
        int c;
        for (int i = 2; i < chars.length; i++) {
            c = chars[i];
            trigramFreq[a][b][c] +=1;
            a = b;
            b = c;
        }
        double s;
        for (int i = 0; i < 52; i++) {
            for (int j = 0; j < 52; j++) {
                s = 0;
                for (int k = 0; k < 52; k++) {
                    trigramFreq[i][j][k] += 1;
                    s += trigramFreq[i][j][k];
                }
                if (normalized) {
                    for (int k = 0; k < 52; k++) {
                        trigramFreq[i][j][k] /= s;
                    }
                }
            }
        }
        return trigramFreq;
    }

    public static int[] textProcess(String path, int n_skip, int max) throws IOException {
        BufferedReader rd = new BufferedReader(new FileReader(path));
        for (int i = 0; i < n_skip; i++) {
            rd.read();
        }
        int c = rd.read();
        int d = -1;
        List<Integer> list = new ArrayList<>();
        int iter = 0;
        boolean added = false;

        while (c != -1 && iter < max) {
            added = false;

            if (Character.isAlphabetic(c)) {
                list.add(toInt(c));
                added = true;
            }

            // handle space
            else {
                d = rd.read();
                if (Character.isAlphabetic(d)){
                    list.add(toInt(d)+26);
                    added = true;
                }
            }

            c = rd.read();
            if (added) iter ++;
        }
        int[] exists = new int[52];

        int[] res = new int[list.size()];
        for (int i = 0; i < res.length; i++) {
            res[i] = list.get(i);
            exists[res[i]] = 1;
        }
        return res;
    }

    public static int toInt(int c){
        if (Character.isUpperCase(c)){
            return c - 65;
        } else {
            return c - 97;
        }
    }

    public static void main(String[] args) throws IOException {
        System.out.println(100000000 < Integer.MAX_VALUE);
        int[] text = textProcess("brown_nolines.txt", 0, Integer.MAX_VALUE);
        int[] arr = Arrays.copyOfRange(text, 0, 100000);
        for (int c : arr) {
            if (c < 26){
                System.out.print(Character.toChars(c+65));
            }
            else{
                System.out.print("_");
                System.out.print(Character.toChars(c-26+65));
            }
        }
    }

    public static int[] read408(String dir) throws FileNotFoundException {
        Scanner rd = new Scanner(new BufferedReader(new FileReader(dir)));
        List<Integer> l = new ArrayList();
        while (rd.hasNext()) {
            l.add(rd.nextInt()-1);
        }
        int[] res = new int[l.size()];
        for (int i = 0; i < l.size(); i++) {
            res[i] = l.get(i);
        }
        return res;
    }

    //read a plain text of 408 I found on the web

    public static int[] plain(String dir) throws IOException {
        BufferedReader rd = new BufferedReader(new FileReader(dir));
        int c = rd.read();
        List<Integer> l = new ArrayList<>();
        while (c != -1) {
            if (Character.isAlphabetic(c)) {
                if (Character.isUpperCase(c)) {
                    l.add(c - 65);
                }
            }
            c = rd.read();
        }
        int[] res = new int[l.size()];
        for (int i = 0; i < l.size(); i++) {
            res[i] = l.get(i);
        }
        return res;
    }
}
