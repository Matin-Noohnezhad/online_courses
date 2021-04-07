<!--<!DOCTYPE html>-->
<!--<html lang="fa" dir="rtl">-->
<!--<head>-->
<!--    <meta charset="UTF-8">-->
<!--    <title>صفحه لاگین</title>-->
<!--    <link rel="stylesheet" href="../css/globalstyle.css">-->
<!--</head>-->
<!--<body>-->
<!--<ul>-->
<!--    <li><a href="../index.php"> صفحه اصلی</a></li>-->
<!--    <li><a href="register.php">صفحه ثبت نام</a></li>-->
<!--    <li><a href="#">صفحه لاگین</a></li>-->
<!--</ul>-->
<!---->
<!--<form method="post" action="login.php">-->
<!--    <fieldset>-->
<!--        <p class="p1">نام کاربری:</p>-->
<!--        <input type="text" name="username">-->
<!--        <br>-->
<!--        <p class="p1">رمز عبور:</p>-->
<!--        <input type="password" name="password">-->
<!--        <br>-->
<!--        <input type="submit" value="ورود">-->
<!--    </fieldset>-->
<!---->
<!---->
<!--</form>-->
<!---->
<!--</body>-->
<!--</html>-->
<?php
$username = $password = "";
$userError = $passError = "";
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $username = test_input($_POST["username"]);
    $password = test_input($_POST["password"]);


    if ($username != "" && $password != "") {
        $servername = "localhost";
        $user = "root";
        $pass = "";
        $dbname = "assignment_3";

// Create connection
        $conn = new mysqli($servername, $user, $pass, $dbname);
        mysqli_set_charset($conn, "utf8");
// Check connection
        if ($conn->connect_error) {
            die("Connection failed: " . $conn->connect_error);
        }

        $sql = "SELECT username, password,admin FROM userpass";
        $result = $conn->query($sql);
        $userFound = false;
        if ($result->num_rows > 0) {
            // output data of each row
            while ($row = $result->fetch_assoc()) {
                if ($row["username"] == $username) {
                    $userFound = true;
                    if ($row["password"] == $password) {
                        //sign in is complete
                        if ($row["admin"] == 1) {
                            session_start();
                            $_SESSION["login"] = 'admin';
                            header("location:admin.php");
                        }else{
                            session_start();
                            $_SESSION["login"] = $username;
                            header("location:../index.php");
                        }
                        //do something here
                    } else {
                        $passError = "پسوورد وارد شده اشتباه است.";
                    }
                    break;
                }
            }
        }
        if ($userFound == false) {
            $userError = "نام کاربری وارد شده در لیست وجود ندارد.";
        }

        //
        $username = "";
        $password = "";
    } else {
        if ($username == "") {
            $userError = "این فیلد را پر کنید*";
        }
        if ($password == "") {
            $passError = "این فیلد را پر کنید*";
        }
    }

}


$string = "<!DOCTYPE html>
<html lang=\"fa\" dir=\"rtl\">
<head>
    <meta charset=\"UTF-8\">
    <title>صفحه لاگین</title>
    <link rel=\"stylesheet\" href=\"../css/globalstyle.css\">
</head>
<body>
<ul>
    <li><a href=\"../index.php\"> صفحه اصلی</a></li>
    <li><a href=\"register.php\">صفحه ثبت نام</a></li>
    <li><a href=\"#\">صفحه لاگین</a></li>
</ul>

<form method=\"post\" action=\"login.php\">
    <fieldset>
        <p class=\"p1\">نام کاربری:</p>
        <input type=\"text\" name=\"username\" value='" . $username . "'>
        <p class=\"p2\" id=\"err1\">" . $userError . "</p>
        <br>
        <p class=\"p1\">رمز عبور:</p>
        <input type=\"password\" name=\"password\" value='" . $password . "'>
        <p class=\"p2\" id=\"err2\">" . $passError . "</p>
        <br>
        <input type=\"submit\" value=\"ورود\">
    </fieldset>


</form>

</body>
</html>";

echo $string;

function test_input($data)
{
    $data = trim($data);
    $data = stripslashes($data);
    $data = htmlspecialchars($data);
    return $data;
}

?>