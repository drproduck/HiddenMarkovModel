import java.io.BufferedWriter;
import java.io.FileWriter;

public class MakeCipher {
    public static void main(String[] args) throws Exception{
        String dir = "/home/drproduck/Documents/HMM/brown_nolines.txt";
        BufferedWriter writer;
        int[][] init = {{1000, 500, 2134}, {10000, 400, 3211}, {100000, 300,1234}};
        for (int i = 0; i < init.length; i++) {
            int[] plainText = util.textProcess(dir, true, init[i][0], init[i][1]);
            int[] dict = util.permuteKey("random", init[i][2]);
            int[] cipher = new int[plainText.length];
            for (int j = 0; j < plainText.length; j++) {
                cipher[j] = dict[plainText[j]];
            }
            writer = new BufferedWriter(new FileWriter(String.format("simple_cipher_length=%d", init[i][1])));
            StringBuilder sb = new StringBuilder();
            for (int j = 0; j < plainText.length; j++) {
                sb.append(plainText[j] + " ");
            }
            sb.append("\n");
            for (int j = 0; j < plainText.length; j++) {
                sb.append(cipher[j] + " ");
            }
            sb.append("\n");
            for (int j = 0; j < plainText.length; j++) {
                sb.append(Character.toString((char) (plainText[j] + 65)));
            }
            sb.append("\n");
            for (int j = 0; j < plainText.length; j++) {
                sb.append(Character.toString((char) (cipher[j] + 65)));
            }
            sb.append("\n");
            writer.write(sb.toString());
            writer.close();
        }

    }
}
