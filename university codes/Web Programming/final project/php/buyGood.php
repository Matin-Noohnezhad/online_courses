<?php
session_start();
if (isset($_SESSION['email']) && !empty($_SESSION['email'])) {
    $signed_in = true;
} else {
    $signed_in = false;
    header("location:../index.php");
}
//
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $good_id = $_POST["good_id"];
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

    $name = '';
    $image_dir = '';
    $categoryName = '';
    $price = 0;
    $store_id = 0;
    $description = '';

    if ($signed_in) {
        $person_id = 0;

        $sql = "SELECT name,price,categoryName,store_id,description,image_dir FROM good where id=" . $good_id;
        $result = $conn->query($sql);

        if ($result->num_rows > 0) {
            // output data of each row
            while ($row = $result->fetch_assoc()) {
                $name = $row["name"];
                $price = $row["price"];
                $categoryName = $row["categoryName"];
                $store_id = $row["store_id"];
                $description = $row["description"];
                $image_dir = $row["image_dir"];


            }
        }
    }
}
$store_name = '';
$sql = "SELECT name FROM store where id=" . $store_id;
$result = $conn->query($sql);
if ($result->num_rows > 0) {
// output data of each row
    while ($row = $result->fetch_assoc()) {
        $store_name = $row["name"];
    }
}
$conn->close();
//
$string = '<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
    <meta charset="UTF-8">
    <title>تقاضای ثبت محصول</title>
    <link rel="stylesheet" href="../css/confirm.css">
</head>
<body>
<div class="main">
    <div class="image">
        <img src="' . $image_dir . '">
    </div>
    <div class="center">
        <br>
        نام محصول:
        ' . $name . '
        <br>
        <br>
        قیمت محصول:
        ' . $price . '
        تومان
        <br>
        <br>
        دسته بندی:
        ' . $categoryName . '
        <br>
        <br>
        نام فروشگاه:
        ' . $store_name . '
        <br>
        <br>
        توضیحات:
        ' . $description . '
    </div>
    <div class="left">
        <br>
        <br>
        <br>
        <form method="post" action="./addGood.php" name="myForm" >
            <input type="number" name="good_id" value=' . $good_id . ' style="display: none">
            <button class="noConfirmBtn"> "اضافه کردن به سبد خرید</button>
        </form>
        <br>
        <br>
        <br>
        <br>
        <br>
    </div>
</div>
<script src="../js/confirm.js"></script>
</body>
</html>';

echo $string;


?>
