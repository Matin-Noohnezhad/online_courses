<?php
$error = '';

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

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $categoryName = test_input($_POST["categoryNames"]);
    $newName = test_input($_POST["newName"]);

    $exist = false;
    $sql = "SELECT categoryName FROM categories";
    $result = $conn->query($sql);

    if ($result->num_rows > 0) {
        // output data of each row
        while ($row = $result->fetch_assoc()) {
            if ($row["categoryName"] == $newName) {
                $exist = true;
                break;
            }
        }
    }
    if ($exist) {
        $error = 'این نام از قبل برای دسته ای دیگر ثبت شده است.';
    } else {
        $sql = "UPDATE categories 
SET 
    categoryName = \"".$newName."\"
WHERE
    categoryName = \"" . $categoryName . "\";";
//        echo $sql;
        $result = $conn->query($sql);


    }
}
$options = '';
$sql2 = "SELECT categoryName FROM categories";
$result2 = $conn->query($sql2);

if ($result2->num_rows > 0) {
    // output data of each row
    while ($row2 = $result2->fetch_assoc()) {
//        $row["name"]
//        <option value="volvo">Volvo</option>
        $options .= '<option value="' . $row2["categoryName"] . '">' . $row2["categoryName"] . '</option>';
    }
}


$conn->close();

$string = '<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
    <meta charset="UTF-8">
    <title>اصلاح نام دسته بندی توسط ادمین</title>
    <link rel="stylesheet" href="../css/update_category.css">
</head>
<body>
<form method="post" action="./update_category.php" name="myForm" onsubmit="return formValidation()">
    <fieldset>
        <legend>اصلاح نام دسته</legend>
        <br>
        انتخاب دسته:
        <br>
        <select id="select1" name="categoryNames" required>
            ' . $options . '
        </select>
        <br>
        <br>
        نام جدید دسته:
        <br>
        <input type="text" name="newName" id="newName">
        <p id="err1">' . $error . '</p>
        <br>
        <br>
        <button>اعمال تغییر</button>
    </fieldset>
</form>

<script src="../js/update_category.js"></script>
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