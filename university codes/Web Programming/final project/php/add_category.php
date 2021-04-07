<?php

$newCategory = "";
$userError = "";
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $newCategory = test_input($_POST["newCategory"]);

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
    $sql = "SELECT categoryName FROM categories";
    $result = $conn->query($sql);
    $oldCategories = "";

    if ($result->num_rows > 0) {
        // output data of each row
        while ($row = $result->fetch_assoc()) {
            $oldCategories .= "<p>";
            $oldCategories .= $row["categoryName"];
            $oldCategories .= "</p>";
            if ($row["categoryName"] == $newCategory) {
                $exist = true;
//                break;
            }
        }
    }
    if ($exist != true) {
        $sql = "INSERT INTO categories (categoryName)
VALUES ('" . $newCategory . "')";

        if ($conn->query($sql) === TRUE) {
//                    echo "New record created successfully";
            $oldCategories .= "<p>";
            $oldCategories .= $newCategory;
            $oldCategories .= "</p>";
            $newCategory = "";
        } else {
            echo "Error: " . $sql . "<br>" . $conn->error;
        }
    }
    if ($exist) {
        $userError = "نام این دسته از قبل در سیستم وجود دارد.*";
    }
    $conn->close();
} else {
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
    $sql = "SELECT categoryName FROM categories";
    $result = $conn->query($sql);
    $oldCategories = "";

    if ($result->num_rows > 0) {
        // output data of each row
        while ($row = $result->fetch_assoc()) {
            $oldCategories .= "<p>";
            $oldCategories .= $row["categoryName"];
            $oldCategories .= "</p>";
        }
    }


}
//

$string = '<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
    <meta charset="UTF-8">
    <title>اضافه کردن دسته توسط ادمین</title>
    <link rel="stylesheet" href="../css/add_to_category.css">
</head>
<body>
<h3>دسته های موجود</h3>
<div>
    ' . $oldCategories . '
</div>
<form method="post" action="add_category.php" id="myForm" onsubmit="return addCategory()">
    <fieldset>
        <legend>اضافه کردن دسته</legend>
        <br>نام دسته جدید:
        <br>
        <input id="newCategory" type="text" name="newCategory" value="' . $newCategory . '">
        <p id="error">' . $userError . '</p>
        <br>
        <br>
        <input id="submit" type="submit" value="ثبت دسته جدید">
    </fieldset>
</form>
<script src="../js/add_to_category.js"></script>
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
