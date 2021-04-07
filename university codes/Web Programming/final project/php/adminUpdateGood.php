<?php
session_start();
session_unset();
session_destroy();

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

$innerBody = '';
$sql = "SELECT id,name,price FROM good";
$result = $conn->query($sql);

if ($result->num_rows > 0) {
    // output data of each row
    while ($row = $result->fetch_assoc()) {
        $innerBody.='<form method="post" action="../php/update_good.php" name="myForm'.$row["id"].'" onsubmit="return formValidation('.$row["id"].')">
    <fieldset>
        <legend>اصلاح اطلاعات محصول</legend>
        <input type="number" name="good_id" value='.$row["id"].' style="display: none">
        <input type="number" name="delete" value=0 style="display: none">
        نام محصول:
        '.$row["name"].'
        <br>
        <br>
        قیمت محصول:
        '.$row["price"].'
        تومان
        <br>
        <br>
        نام جدید محصول:
        <br>
        <input type="text" name="newName">
        <br>
        <p id="err'.$row["id"].'"></p>
        <br>
        قیمت جدید محصول:
        <br>
        <input type="number" min="0" max="1000000000" step="1" name="newPrice">
        <br>
        <br>
        <button onclick="setDelete('.$row["id"].')">حذف محصول</button>
        <button style="margin-left: 5px;">اعمال تغییرات</button>

    </fieldset>
</form>';

    }
}

$conn->close();

$string = '<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
    <meta charset="UTF-8">
    <title>اصلاح اطلاعات محصول</title>
    <link rel="stylesheet" href="../css/update_category.css">
</head>
<body>

'.$innerBody.'
<script src="../js/sellerUpdateGoods.js"></script>
</body>
</html>';

echo $string;

?>
