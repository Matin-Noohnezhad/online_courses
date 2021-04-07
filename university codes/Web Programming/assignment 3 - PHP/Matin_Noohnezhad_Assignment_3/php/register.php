<!--<!DOCTYPE html>-->
<!--<html lang="fa" dir="rtl">-->
<!--<head>-->
<!--    <meta charset="UTF-8">-->
<!--    <title>صفحه ثبت نام</title>-->
<!--    <link rel="stylesheet" href="../css/globalstyle.css">-->
<!--</head>-->
<!--<body>-->
<!--<ul>-->
<!--    <li><a href="../index.php"> صفحه اصلی</a></li>-->
<!--    <li><a href="#">صفحه ثبت نام</a></li>-->
<!--    <li><a href="login.php">صفحه لاگین</a></li>-->
<!--</ul>-->
<!---->
<!--<form method="post" action="register.php">-->
<!--    <fieldset>-->
<!--        <p class="p1">نام کاربری:</p>-->
<!--        <input type="text" name="username">-->
<!--        <p class="p2" id="err1"></p>-->
<!--        <br>-->
<!--        <p class="p1">رمز عبور:</p>-->
<!--        <input type="password" name="pass">-->
<!--        <p class="p2" id="err2"></p>-->
<!--        <br>-->
<!--        <p class="p1">تکرار رمز عبور:</p>-->
<!--        <input type="password" name="repass">-->
<!--        <p class="p2" id="err3"></p>-->
<!--        <br>-->
<!--        <input type="submit" value="ثبت نام" name="register">-->
<!--    </fieldset>-->
<!--</form>-->
<!---->
<!--</body>-->
<!--</html>-->
<?php
// define variables and set to empty values
$username = $pass = $repass = "";
$userError = $passError = $repassError = "";
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $username = test_input($_POST["username"]);
    $pass = test_input($_POST["pass"]);
    $repass = test_input($_POST["repass"]);

    if ($username == "")
        $userError = "این فیلد را پر کنید*";
    if ($pass == "")
        $passError = "این فیلد را پر کنید*";
    if ($repass == "")
        $repassError = "این فیلد را پر کنید*";
    //
    if ($username != "" && $pass != "" && $repass != "") {
        $repeat = false;
        if ($pass == $repass) {

            $servername = "localhost";
            $user = "root";
            $password = "";
            $dbname = "assignment_3";

// Create connection
            $conn = new mysqli($servername, $user, $password, $dbname);
            mysqli_set_charset($conn, "utf8");
// Check connection
            if ($conn->connect_error) {
                die("Connection failed: " . $conn->connect_error);
            }

            $sql = "SELECT username, password FROM userpass";
            $result = $conn->query($sql);

            if ($result->num_rows > 0) {
                // output data of each row
                while ($row = $result->fetch_assoc()) {
                    if ($row["username"] == $username) {
                        $repeat = true;
                        break;
                    }
                }
            }
            if ($repeat != true) {
                $sql = "INSERT INTO userpass (username, password,admin)
VALUES ('".$username."','".$pass."' , 0)";

                if ($conn->query($sql) === TRUE) {
//                    echo "New record created successfully";
                } else {
                    echo "Error: " . $sql . "<br>" . $conn->error;
                }
            }
            if($repeat){
                $userError = "این یوزرنیم تکراری است.*";
            }
            $conn->close();
        }
        //
        $username = $pass = $repass = "";
    }
}
$string = "<!DOCTYPE html>
<html lang=\"fa\" dir=\"rtl\">
<head>
    <meta charset=\"UTF-8\">
    <title>صفحه ثبت نام</title>
    <link rel=\"stylesheet\" href=\"../css/globalstyle.css\">
</head>
<body>
<ul>
    <li><a href=\"../index.php\"> صفحه اصلی</a></li>
    <li><a href=\"#\">صفحه ثبت نام</a></li>
    <li><a href=\"login.php\">صفحه لاگین</a></li>
</ul>

<form method=\"post\" action=\"register.php\">
    <fieldset>
        <p class=\"p1\">نام کاربری:</p>
        <input type=\"text\" name=\"username\" value=\"" . $username . " \">
        <p class=\"p2\" id=\"err1\">" . $userError . "</p>
        <br>
        <p class=\"p1\">رمز عبور:</p>
        <input type=\"password\" name=\"pass\" value='" . $pass . "'>
        <p class=\"p2\" id=\"err2\">" . $passError . "</p>
        <br>
        <p class=\"p1\">تکرار رمز عبور:</p>
        <input type=\"password\" name=\"repass\" value='" . $repass . "'>
        <p class=\"p2\" id=\"err3\">" . $repassError . "</p>
        <br>
        <input type=\"submit\" value=\"ثبت نام\" name=\"register\">
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

