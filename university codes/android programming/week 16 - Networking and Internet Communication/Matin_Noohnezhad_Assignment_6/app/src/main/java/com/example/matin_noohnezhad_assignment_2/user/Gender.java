package com.example.matin_noohnezhad_assignment_2.user;

public enum Gender {
    MALE, FEMALE;

    public static Gender toGender(String gender) {
        switch (gender) {
            case "Female":
                return FEMALE;
            case "Male":
                return MALE;
        }
        return null;
    }


}
