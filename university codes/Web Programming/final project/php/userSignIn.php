<?php
session_start();
//if (isset($_SESSION['username']) && !empty($_SESSION['username'])) {
//    echo 'matin';
//}

$email = $pass = "";
$userError1 = $userError2 = "";
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $email = test_input($_POST["email"]);
    $pass = test_input($_POST["pass"]);

    $servername = "localhost";
    $user = "root";
    $password = "";
    $dbname = "final_project";

// Create connection
    $conn = new mysqli($servername, $user, $password, $dbname);
    mysqli_set_charset($conn, "utf8");
// Check connection
    if ($conn->connect_error) {
        die("Connection failed: " . $conn->connect_error);
    }

    $emailExist = false;
    $passCorrect = true;
    $admin = false;
    $username = '';
    $sql = "SELECT email,password,isAdmin,username FROM person";
    $result = $conn->query($sql);

    if ($result->num_rows > 0) {
        // output data of each row
        while ($row = $result->fetch_assoc()) {
            if ($row["email"] == $email) {
                $emailExist = true;
                if (!($row["password"] == $pass)) {
                    $passCorrect = false;
                } elseif ($row["isAdmin"] == 1) {
                    $admin = true;
                } else {
                    $username = $row["username"];
                }

                break;
            }
        }
    }
    if ($emailExist && $passCorrect) {
        if ($admin) {
            header("location:./admin.php");
        } else {
            $_SESSION['username'] = $username;
            $_SESSION['email'] = $email;
            header("location:../index.php");
        }
    } else {
        if (!$emailExist) {
            $userError1 = 'ایمیل وارد شده در سیستم وجود ندارد.';
        } else {
            $userError2 = 'رمز عبور وارد شده با ایمیل همخوانی ندارد.';
            $pass = '';
        }
    }
    $conn->close();
}

$string = '<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
    <meta charset="UTF-8">
    <title>ورود به حساب کاربری</title>
    <link rel="stylesheet" href="../css/userRegister.css">
</head>
<body>
<form method="post" action="userSignIn.php" name="myForm" onsubmit="return formValidation()">
    <fieldset>
        <legend>ثبت نام کاربر</legend>
        <br>
        ایمیل:
        <br>
        <input class="input" type="text" name="email" value="' . $email . '">
        <p class="error" id="err1">' . $userError1 . '</p>
        <br>
        <br>
        رمز عبور:
        <br>
        <input class="input" type="password" name="pass" value="' . $pass . '">
        <p class="error" id="err2">' . $userError2 . '</p>
        <br>
        <br>
        <input id="submit" type="submit" value="ورود">
    </fieldset>
</form>

<script src="../js/userSignIn.js"></script>
</body>
</html>';

echo $string;

function test_input($data)
{
    $data = trim($data);
    $data = stripslashes($data);
    $data = htmlspecialchars($data);
    return $data;
}

?>