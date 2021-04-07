public class LanguageAcceptor {

    public static void main(String[] args) {
        String sentence = "I wanna, go out at 2 pm.";
        System.out.println(q0(sentence, 0));
    }

    public static boolean q0(String sentence, int index) {
        if (sentence.length() == index)
            return false;
        if (sentence.charAt(index) == 73)
            return q5(sentence, index + 1);
        if (sentence.charAt(index) > 64 && sentence.charAt(index) < 91)
            return q4(sentence, index + 1);
        return false;
    }

    public static boolean q1(String sentence, int index) {
        if (sentence.length() == index)
            return false;
        if (sentence.charAt(index) == 46)
            return q7(sentence, index + 1);
        if (sentence.charAt(index) > 47 && sentence.charAt(index) < 58)
            return q1(sentence, index + 1);
        if (sentence.charAt(index) == 32)
            return q2(sentence, index + 1);
        if (sentence.charAt(index) == 44 || sentence.charAt(index) == 59)
            return q3(sentence, index + 1);
        return false;
    }

    public static boolean q2(String sentence, int index) {
        if (sentence.length() == index)
            return false;
        if (sentence.charAt(index) > 47 && sentence.charAt(index) < 58)
            return q1(sentence, index + 1);
        if (sentence.charAt(index) == 73)
            return q5(sentence, index + 1);
        if (sentence.charAt(index) == 97)
            return q6(sentence, index + 1);
        if ((sentence.charAt(index) > 64 && sentence.charAt(index) < 91) || (sentence.charAt(index) > 96 && sentence.charAt(index) < 123))
            return q4(sentence, index + 1);
        return false;
    }

    public static boolean q3(String sentence, int index) {
        if (sentence.length() == index)
            return false;
        if (sentence.charAt(index) == 32)
            return q8(sentence, index + 1);
        return false;
    }

    public static boolean q4(String sentence, int index) {
        if (sentence.length() == index)
            return false;
        if ((sentence.charAt(index) > 64 && sentence.charAt(index) < 91) || (sentence.charAt(index) > 96 && sentence.charAt(index) < 123))
            return q9(sentence, index + 1);
        return false;
    }

    public static boolean q5(String sentence, int index) {
        if (sentence.length() == index)
            return false;
        if (sentence.charAt(index) == 32)
            return q2(sentence, index + 1);
        if (sentence.charAt(index) == 44 || sentence.charAt(index) == 59)
            return q3(sentence, index + 1);
        if (sentence.charAt(index) == 46)
            return q11(sentence, index + 1);
        if ((sentence.charAt(index) > 64 && sentence.charAt(index) < 91) || (sentence.charAt(index) > 96 && sentence.charAt(index) < 123))
            return q9(sentence, index + 1);
        return false;
    }

    public static boolean q6(String sentence, int index) {
        if (sentence.length() == index)
            return false;
        if (sentence.charAt(index) == 32)
            return q2(sentence, index + 1);
        if (sentence.charAt(index) == 44 || sentence.charAt(index) == 59)
            return q3(sentence, index + 1);
        if (sentence.charAt(index) == 46)
            return q11(sentence, index + 1);
        if ((sentence.charAt(index) > 64 && sentence.charAt(index) < 91) || (sentence.charAt(index) > 96 && sentence.charAt(index) < 123))
            return q9(sentence, index + 1);
        return false;
    }

    public static boolean q7(String sentence, int index) {
        if (sentence.length() == index)
            return true;
        if (sentence.charAt(index) > 47 && sentence.charAt(index) < 58)
            return q10(sentence, index + 1);
        return false;
    }

    public static boolean q8(String sentence, int index) {
        if (sentence.length() == index)
            return false;
        if (sentence.charAt(index) == 73)
            return q5(sentence, index + 1);
        if (sentence.charAt(index) == 97)
            return q6(sentence, index + 1);
        if ((sentence.charAt(index) > 64 && sentence.charAt(index) < 91) || (sentence.charAt(index) > 96 && sentence.charAt(index) < 123))
            return q4(sentence, index + 1);
        return false;
    }

    public static boolean q9(String sentence, int index) {
        if (sentence.length() == index)
            return false;
        if ((sentence.charAt(index) > 64 && sentence.charAt(index) < 91) || (sentence.charAt(index) > 96 && sentence.charAt(index) < 123))
            return q9(sentence, index + 1);
        if (sentence.charAt(index) == 32)
            return q2(sentence, index + 1);
        if (sentence.charAt(index) == 44 || sentence.charAt(index) == 59)
            return q3(sentence, index + 1);
        if (sentence.charAt(index) == 46)
            return q11(sentence, index + 1);
        return false;
    }

    public static boolean q10(String sentence, int index) {
        if (sentence.length() == index)
            return false;
        if (sentence.charAt(index) > 47 && sentence.charAt(index) < 58)
            return q10(sentence, index + 1);
        if (sentence.charAt(index) == 46)
            return q11(sentence, index + 1);
        if (sentence.charAt(index) == 32)
            return q2(sentence, index + 1);
        if (sentence.charAt(index) == 44 || sentence.charAt(index) == 59)
            return q3(sentence, index + 1);
        return false;
    }

    public static boolean q11(String sentence, int index) {
        if (sentence.length() == index)
            return true;
        return false;
    }

}
