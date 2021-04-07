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
<!--    <li><a href="login.php">صفحه لاگین</a></li>-->
<!--</ul>-->
<!---->
<!--<form method="post" action="admin.php">-->
<!--    <fieldset>-->
<!--        <p class="p1">عنوان مطلب:</p>-->
<!--        <input type="text" name="title">-->
<!--        <br>-->
<!--        <p class="p1">متن:</p>-->
<!--        <br>-->
<!--        <textarea name="content" rows="10" cols="50"></textarea>-->
<!--        <br>-->
<!--        <input type="submit" value="ذخیره مطلب">-->
<!--    </fieldset>-->
<!--</form>-->
<!---->
<!--</body>-->
<!--</html>-->
<?php
$title = $content = "";
$titleError = $contentError = "";
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $title = test_input($_POST["title"]);
    $content = test_input($_POST["content"]);
    if ($title != "" && $content != "") {
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
        $sql = "INSERT INTO data (title, content)
VALUES ('" . $title . "' , '" . $content . "')";

        if ($conn->query($sql) === TRUE) {
//            echo "New record created successfully";
        } else {
            echo "Error: " . $sql . "<br>" . $conn->error;
        }


        $title = "";
        $content = "";

    } else {
        if ($title == "") {
            $titleError = "این فیلد را پر کنید*";
        }
        if ($content == "") {
            $contentError = "این فیلد را پر کنید*";
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
    <li><a href=\"login.php\">صفحه لاگین</a></li>
</ul>

<form method=\"post\" action=\"admin.php\">
    <fieldset>
        <p class=\"p1\">عنوان مطلب:</p>
        <input type=\"text\" name=\"title\" value='" . $title . "'>
        <p class=\"p2\" id=\"err1\">" . $titleError . "</p>
        <br>
        <p class=\"p1\">متن:</p>
        <br>
        <textarea name=\"content\" rows=\"10\" cols=\"50\">" . $content . "</textarea>
        <p class=\"p2\" id=\"err2\">" . $contentError . "</p>
        <br>
        <input type=\"submit\" value=\"ذخیره مطلب\">
    </fieldset>
</form>

</body>
</html>";

session_start();
if (isset($_SESSION["login"])) {
    if ($_SESSION['login'] == 'admin') {
        echo $string;
    } else {
        die("you dont have the permission to come to this page!!!");
    }
} else {
    die("you dont have the permission to come to this page!!!");
}
function test_input($data)
{
    $data = trim($data);
    $data = stripslashes($data);
    $data = htmlspecialchars($data);
    return $data;
}

?>
