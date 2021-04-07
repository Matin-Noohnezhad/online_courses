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
    $sql = "SELECT email FROM store";
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
        $sql = "INSERT INTO store (name, password,email)
VALUES ('" . $username . "','" . $pass . "','" . $email . "')";

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
    <title>ثبت نام فروشگاه</title>
    <link rel="stylesheet" href="../css/sellerRegister.css">
</head>
<body>
<form method="post" action="sellerRegister.php" name="myForm" onsubmit="return formValidation()">
    <fieldset>
        <legend>ثبت نام فروشگاه</legend>
        <br>
        ایمیل:
        <br>
<!--        <input class="input" type="email" name="email">-->
        <input class="input" type="text" name="email">
        <p class="error" id="err1">' . $userError . '</p>
        <br>
        <br>
        نام فروشگاه:
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
        <a href="sellerSignIn.php">ورود به حساب فروشگاه</a>
        
    </fieldset>
</form>


<script src="../js/seller_register.js"></script>
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
