<?php
$email = $username = $pass = "";
$userError = "";
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $email = test_input($_POST["email"]);
    $username = test_input($_POST["username"]);
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

    $exist = false;
    $sql = "SELECT email FROM person";
    $result = $conn->query($sql);

    if ($result->num_rows > 0) {
        // output data of each row
        while ($row = $result->fetch_assoc()) {
            if ($row["email"] == $email) {
                $exist = true;
                break;
            }
        }
    }
    if ($exist != true) {
        $sql = "INSERT INTO person (username, password,email,isAdmin)
VALUES ('" . $username . "','" . $pass . "','" . $email . "' , 0)";

        if ($conn->query($sql) === TRUE) {
//                    echo "New record created successfully";
            $username = "";
            $pass = "";
        } else {
            echo "Error: " . $sql . "<br>" . $conn->error;
        }
    }
    if ($exist) {
        $email = "";
        $userError = "این ایمیل در سیستم وجود دارد.*";
    }
    $conn->close();
}
//


$string = '<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
    <meta charset="UTF-8">
    <title>ثبت نام کاربر</title>
    <link rel="stylesheet" href="../css/userRegister.css">
</head>
<body>
<form method="post" action="register.php" name="myForm" onsubmit="return formValidation()">
    <fieldset>
        <legend>ثبت نام کاربر</legend>
        <br>
        ایمیل:
        <br>
<!--        <input class="input" type="email" name="email">-->
        <input class="input" type="text" name="email">
        <p class="error" id="err1">' . $userError . '</p>
        <br>
        <br>
        نام کاربری:
        <br>
        <input class="input" type="text" name="username" value="' . $username . '">
        <p class="error" id="err2"></p>
        <br>
        <br>
        رمز عبور:
        <br>
        <input class="input" type="password" name="pass" value="' . $pass . '">
        <p class="error" id="err3"></p>
        <br>
        <br>
        تکرار رمز عبور:
        <br>
        <input class="input" type="password" name="repass" value="' . $pass . '">
        <p class="error" id="err4"></p>
        <br>
        <br>
        <input id="submit" type="submit" value="ثبت">
        <br>
        <a href="userSignIn.php">ورود به حساب کاربری</a>
    </fieldset>
</form>


<script src="../js/user_register.js"></script>
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